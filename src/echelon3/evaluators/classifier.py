import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import shutil  # добавлено для копирования исходных файлов
import cv2
import numpy as np

from echelon3.evaluators.basic import Evaluator
from echelon3.data.imageclassifier import FoldersHiveImageClassifierDataset


class ClassifierEvaluator(Evaluator):
    """Оценка классификатора по валидационному (test) датасету
    + сохранение ошибок в папки X внутри scores_and_labels.

    В каталоге X хранятся сэмплы, где истинный класс X,
    а модель на них ошибается.

    Конфиг evaluator.config должен содержать:
      - scores_and_labels: str  # базовая папка для сохранения ошибок
    """

    def __init__(
        self,
        net,
        dataloader: Optional[DataLoader],
        metric,
        preprocess,
        postprocess,
        scores_and_labels: str,
        **kwargs,
    ):
        super().__init__(
            net=net,
            dataloader=dataloader,
            metric=metric,
            preprocess=preprocess,
            postprocess=postprocess,
            **kwargs,
        )

        self.errors_root = scores_and_labels
        os.makedirs(self.errors_root, exist_ok=True)

        # переносим метрику на тот же девайс, что и сеть (если поддерживает to)
        if hasattr(self.metric, "to"):
            self.metric = self.metric.to(self.device)

    def _build_errors_dataset(
        self,
        error_classes: set[int],
        base_dataset,
    ) -> Optional[FoldersHiveImageClassifierDataset]:
        """
        Собирает датасет по папкам ошибок:
          errors_root/<class_id>/*.png|*.jpg|...

        Использует ТЕ ЖЕ augment и preprocess, что и исходный тестовый датасет,
        чтобы пайплайн полностью совпадал с основным валидатором.
        """
        if not error_classes:
            return None

        num_classes = max(error_classes) + 1

        ds = FoldersHiveImageClassifierDataset(
            augment=getattr(base_dataset, "augment", None),
            preprocess=getattr(base_dataset, "preprocess", None),
            label_type="class",
            classes=num_classes,
            folder=self.errors_root,
            wildcards=["*.png", "*.jpg", "*.jpeg", "*.bmp"],
        )
        return ds

    def _evaluate_on_errors(self, batch_size: int, error_classes: set[int], base_dataset):
        """
        Дополнительный прогон: считаем метрику на собранных ошибках.
        Ожидаем, что accuracy (и подобные метрики) будут равны 0
        (т.к. мы собрали только неправильно классифицированные примеры).
        """
        errors_dataset = self._build_errors_dataset(error_classes, base_dataset)
        if errors_dataset is None or len(errors_dataset) == 0:
            print("\n--> No error samples collected, skipping secondary evaluation on errors.")
            return

        errors_loader = DataLoader(
            errors_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Сбрасываем метрику и прогоняем по ошибкам
        if hasattr(self.metric, "reset"):
            self.metric.reset()

        self.net.to(self.device)
        self.net.eval()

        with torch.no_grad():
            total_size = len(errors_dataset)
            progress = tqdm(
                initial=0,
                total=total_size,
                desc="--> Evaluating on collected errors",
                ncols=0,
            )

            for images, labels in errors_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.net(images)

                # обновляем метрику
                self.metric.update(logits, labels)

                batch_size = images.size(0)
                progress.update(batch_size)

            progress.close()

        result = self.metric.compute()
        if torch.is_tensor(result) and result.numel() == 1:
            result = float(result.cpu())

        print(f"--> Metric on collected errors: {result}")
        return result

    def evaluate_one(self, dataloader: DataLoader, mode: str = "test"):
        if dataloader is None:
            raise RuntimeError("ClassifierEvaluator: dataloader is None")

        self.net.to(self.device)
        self.net.eval()

        if hasattr(self.metric, "reset"):
            self.metric.reset()

        # глобальный индекс только для fallback-сохранения тензоров
        global_idx = 0
        # проверяем, умеет ли датасет возвращать путь к исходному файлу
        has_source_path = hasattr(dataloader.dataset, "get_source_path")
        # сдвиг индекса в пределах всего датасета (по порядку выдачи батчей)
        dataset_idx_offset = 0

        # для последующей проверки: какие классы имели ошибки
        error_classes: set[int] = set()
        last_batch_size: int = dataloader.batch_size or 1

        with torch.no_grad():
            total_size = len(dataloader.dataset)
            progress = tqdm(
                initial=0,
                total=total_size,
                desc=f"--> Evaluating ({mode}) ",
                ncols=0,
            )

            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.net(images)

                # обновляем метрику на основном датасете
                self.metric.update(logits, labels)

                # вычисление предсказаний
                if logits.ndim > 1 and logits.size(-1) > 1:
                    preds = torch.argmax(logits, dim=-1)
                else:
                    preds = (logits > 0).long().view_as(labels)

                wrong_mask = preds != labels
                if wrong_mask.any():
                    wrong_idx = torch.nonzero(wrong_mask, as_tuple=False).squeeze(1)

                    for wi in wrong_idx:
                        wi_int = int(wi.item())
                        img = images[wi_int].detach().cpu()
                        true_cls = int(labels[wi_int].detach().cpu())
                        pred_cls = int(preds[wi_int].detach().cpu())

                        # запоминаем класс, для которого были ошибки
                        error_classes.add(true_cls)

                        # каталог по ИСТИННОМУ классу: errors_root/<true_cls>
                        subdir = os.path.join(self.errors_root, f"{true_cls}")
                        os.makedirs(subdir, exist_ok=True)

                        # глобальный индекс сэмпла в датасете (по порядку прохода)
                        dataset_global_idx = dataset_idx_offset + wi_int

                        saved = False
                        if has_source_path:
                            try:
                                src_path = dataloader.dataset.get_source_path(
                                    dataset_global_idx
                                )
                            except TypeError:
                                # на случай старой сигнатуры без idx
                                src_path = dataloader.dataset.get_source_path()

                            if src_path is not None and os.path.exists(src_path):
                                # сохраняем именно исходный файл
                                dst_name = os.path.basename(src_path)
                                dst_path = os.path.join(subdir, dst_name)
                                shutil.copy2(src_path, dst_path)
                                saved = True

                        # fallback: если не удалось получить/скопировать исходный файл,
                        # сохраняем преобразованное изображение
                        if not saved:
                            fname = f"{global_idx:08d}.png"
                            fpath = os.path.join(subdir, fname)
                            save_image(img, fpath)
                            global_idx += 1

                batch_size = len(images)
                last_batch_size = batch_size
                dataset_idx_offset += batch_size
                progress.update(batch_size)

            progress.close()

        # основной результат по исходному датасету
        result = self.metric.compute()
        if torch.is_tensor(result) and result.numel() == 1:
            result = float(result.cpu())

        print(f"--> Main metric on validation ({mode}): {result}")

        # дополнительная проверка на собранных ошибках:
        # используем тот же augment/preprocess, что и в исходном датасете
        self._evaluate_on_errors(
            batch_size=last_batch_size,
            error_classes=error_classes,
            base_dataset=dataloader.dataset,
        )

        return result