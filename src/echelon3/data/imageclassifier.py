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
        """Return the path to the source file by dataset index."""
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

        # flat list of files in the order of classes and their files
        self._flat_filenames = []
        for c in range(self.classes):
            self._flat_filenames.extend(self.filenames[c])

    def get_source_path(self, idx: int) -> str | None:
        """Return the path to the source file by *dataset index*.

        IMPORTANT: we use the same indexing logic as get_item/__getitem__,
        so the path corresponds exactly to the sample that goes into the network.
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
    Analogous to FoldersHiveImageClassifierDataset, but:
      * does not use "0/1/..." subfolders,
      * takes ALL files from `folder/**/wildcards`,
      * all samples share the same class `fixed_label`.

    Config:
      module: echelon3.data.imageclassifier
      type: FolderWithFixedLabelDataset
      config:
        folder: '/data2/guardora/celeba-a/train/0'
        fixed_label: 0
        wildcards: ['*.jpg','*.jpeg','*.png']
    """

    folder = None        # path to the folder with files of a SINGLE class
    fixed_label = None   # label of this class (int)
    wildcards = None     # list of masks, as in FoldersHiveImageClassifierDataset

    # We do not override __init__:
    # BasicDataset.__init__ sets folder/fixed_label/wildcards from kwargs
    # and FilesDataset.__init__ calls collect_filenames_with_labels()

    def collect_filenames_with_labels(self):
        """
        Fill self.filenames in the format expected by PerClassFilesDataset:
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

        # flat list for get_source_path, by analogy with FoldersHiveImageClassifierDataset
        self._flat_filenames = list(self.filenames[label])

    def get_source_path(self, idx: int) -> str | None:
        """Return the path to the source file by dataset index.

        We use PerClassFilesDataset.get_item so that
        the index matches what __getitem__ sees.
        """
        try:
            item = self.get_item(idx)  # (path, label)
        except Exception:
            return None

        if isinstance(item, (list, tuple)) and len(item) >= 1:
            return item[0]
        return None
