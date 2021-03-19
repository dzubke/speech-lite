"""
"""
# standard libraries
import os
from pathlib import Path
# third-party libs
from google.cloud import storage

class GCSCheckpointHandler():
    def __init__(self, cfg):
        self.client = storage.Client()
        self.local_save_dir = cfg['local_save_path']
        self.gcs_bucket = cfg['gcs_bucket']
        self.gcs_dir = cfg['gcs_dir']
        self.bucket = self.client.bucket(bucket_name=self.gcs_bucket)
        self.chkpt_per_epoch = cfg['checkpoints_per_epoch']

    def download_from_gcs_bucket(self, filepath:str):
        """
        Finds an object with `filepath` in the gcs bucket. If it exists, the
        object is downloaded to a local file in the tmp-directory, and returns the local file path.
        If the filepath doesn't match the gcs bucket, returns None.
        :return: the local path in tmp-dir to object, or None if no objects are found.
        """
        paths = list(self.client.list_blobs(self.gcs_bucket, prefix=filepath))
        if paths:
            #paths.sort(key=lambda x: x.time_created)
            #latest_blob = paths[-1]
            local_path = os.path.join("/tmp/", filepath)
            os.makedirs(local_path, exist_ok=True)
            paths[0].download_to_filename(local_path)
            return local_path
        else:
            return None

    def _save_model(self, filepath: str, trainer, pl_module):

        # make paths
        if trainer.is_global_zero:
            tqdm.write("Saving model to %s" % filepath)
            trainer.save_checkpoint(filepath)
            self._save_file_to_gcs(filepath)

    def upload_to_gcs(self, filename:str):
        gcs_path = os.path.join(self.gcs_dir, filename)
        local_path = os.path.join(self.local_save_dir, filename)
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)

    def upload_tensorboard_ckpt(self):
        tb_train_dir = os.path.join(self.local_save_dir, "train")
        tb_dev_dir = os.path.join(self.local_save_dir, "dev")

        self._upload_dir_to_gcs(tb_train_dir, "train")
        self._upload_dir_to_gcs(tb_dev_dir, "dev")

    def _upload_dir_to_gcs(self, dir_path:str, gcs_path_prefix:str):
        """This function recursively uploads all of the contents of a directory
        to the gcs bucket and directory
        """
        dir_path = Path(dir_path)
        for local_file in dir_path.glob('*'):
            if local_file.is_dir():
               self._upload_dir_to_gcs(str(local_file), os.path.join(gcs_path_prefix,  local_file.name))
            elif local_file.is_file():
               self.upload_to_gcs(os.path.join(gcs_path_prefix, local_file.name))
            else:
                continue
