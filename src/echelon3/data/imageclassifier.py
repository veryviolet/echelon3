import glob
import os
from random import shuffle
import pandas as pd

from echelon3.data.basic import FilesDataset, AllFilesDataset, PerClassFilesDataset


DATAFRAME_SOURCE_TYPE_CSV = 'csv'


class DataFrameImageClassifierDataset(AllFilesDataset):
    datasource_type: str = None
    datasource: str = None
    separator: str = ','
    filename_column = None
    label_column = None
    classes = None

    def collect_filenames_with_labels(self):
        if self.datasource_type == DATAFRAME_SOURCE_TYPE_CSV:
            try:
                df = pd.read_csv(self.datasource, sep=self.separator, low_memory=False)
                df = df.sample(frac=1).reset_index(drop=True)
                self.filenames_with_labels = list(
                    df[[self.filename_column, self.label_column]].itertuples(index=False)
                )
                self.filenames_with_labels = (
                    [tuple(x) for x in self.filenames_with_labels if x[1] in self.classes]
                    if self.classes is not None
                    else [tuple(x) for x in self.filenames_with_labels]
                )
            except Exception as e:
                raise RuntimeError(f'failed to read csv data: {e}')
        else:
            raise NotImplementedError('that source is not implemented yet')

    def get_source_path(self, idx: int) -> str | None:
        """Вернуть путь к исходному файлу по индексу датасета."""
        if hasattr(self, "filenames_with_labels") and 0 <= idx < len(self.filenames_with_labels):
            return self.filenames_with_labels[idx][0]
        return None


class FoldersHiveImageClassifierDataset(PerClassFilesDataset):

    folder = None
    classes = None
    wildcards = None

    def collect_filenames_with_labels(self):
        self.filenames = {c: [] for c in range(self.classes)}
        for c in range(self.classes):
            oneclass = str(c)
            for wc in self.wildcards:
                for f in glob.glob(os.path.join(self.folder, oneclass, '**', wc), recursive=True):
                    if self.filter is None or self.filter.check_file(f):
                        self.filenames[c].append(f)
                        print(f'\rfiles: 0: {len(self.filenames[0])}, 1: {len(self.filenames[1])}', end='')

            shuffle(self.filenames[c])

        # плоский список файлов в порядке классов и их файлов
        self._flat_filenames = []
        for c in range(self.classes):
            self._flat_filenames.extend(self.filenames[c])

    def get_source_path(self, idx: int) -> str | None:
        """Вернуть путь к исходному файлу по *индексу датасета*.

        ВАЖНО: используем ту же логику индексации, что и get_item/__getitem__,
        чтобы путь соответствовал именно тому сэмплу, который идёт в сеть.
        """
        try:
            item = self.get_item(idx)  # (path, label)
        except Exception:
            return None

        if isinstance(item, (list, tuple)) and len(item) >= 1:
            return item[0]
        return None


class FolderWithFixedLabelDataset(PerClassFilesDataset):
    """
    Аналог FoldersHiveImageClassifierDataset, но:
      * не использует подпапки "0/1/...",
      * берёт ВСЕ файлы из `folder/**/wildcards`,
      * все сэмплы имеют один и тот же класс `fixed_label`.

    Конфиг:
      module: echelon3.data.imageclassifier
      type: FolderWithFixedLabelDataset
      config:
        folder: '/data2/guardora/celeba-a/train/0'
        fixed_label: 0
        wildcards: ['*.jpg','*.jpeg','*.png']
    """

    folder = None        # путь к папке с файлами ОДНОГО класса
    fixed_label = None   # метка этого класса (int)
    wildcards = None     # список масок, как у FoldersHiveImageClassifierDataset

    # __init__ не переопределяем:
    # BasicDataset.__init__ проставит folder/fixed_label/wildcards из kwargs
    # и FilesDataset.__init__ вызовет collect_filenames_with_labels()

    def collect_filenames_with_labels(self):
        """
        Заполняем self.filenames в формате, который ожидает PerClassFilesDataset:
          { fixed_label: [list_of_paths] }
        """
        if self.folder is None:
            raise RuntimeError("FolderWithFixedLabelDataset: 'folder' must be specified")
        if not os.path.isdir(self.folder):
            raise RuntimeError(
                f"FolderWithFixedLabelDataset: folder '{self.folder}' does not exist or is not a directory"
            )

        label = int(self.fixed_label)
        patterns = self.wildcards or ["*.jpg", "*.jpeg", "*.png"]

        self.filenames = {label: []}

        for wc in patterns:
            for f in glob.glob(os.path.join(self.folder, '**', wc), recursive=True):
                if self.filter is None or self.filter.check_file(f):
                    self.filenames[label].append(f)

        shuffle(self.filenames[label])

        # плоский список для get_source_path по аналогии с FoldersHiveImageClassifierDataset
        self._flat_filenames = list(self.filenames[label])

    def get_source_path(self, idx: int) -> str | None:
        """Вернуть путь к исходному файлу по индексу датасета.

        Используем PerClassFilesDataset.get_item, чтобы
        индекс совпадал с тем, что видит __getitem__.
        """
        try:
            item = self.get_item(idx)  # (path, label)
        except Exception:
            return None

        if isinstance(item, (list, tuple)) and len(item) >= 1:
            return item[0]
        return None
