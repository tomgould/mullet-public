"""Agent checkpoint manager: snapshot/restore working-directory state across runs."""
import os
import shutil
import datetime
import threading
import subprocess

class CheckpointManager:
    """
    Manages filesystem-based checkpoints for a given working directory.

    Snapshots are stored in a base directory, organized by timestamp.
    It provides functionality to create, list, restore, and ensure snapshots,
    with automatic cleanup of old snapshots.
    """

    def __init__(self, base_dir='.mullet/checkpoints', max_snapshots=5):
        """
        Initializes the CheckpointManager.

        Args:
            base_dir (str): The base directory where checkpoints will be stored.
                            Defaults to '.mullet/checkpoints'.
            max_snapshots (int): The maximum number of snapshots to retain.
                                 Oldest snapshots are removed when this limit is exceeded.
                                 Defaults to 5.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.max_snapshots = max_snapshots
        os.makedirs(self.base_dir, exist_ok=True)
        self._lock = threading.RLock()

    def _get_snapshot_path(self, timestamp):
        """Helper to get the full path for a snapshot."""
        return os.path.join(self.base_dir, timestamp)

    def _get_tracked_files(self, work_dir):
        """
        Determines which files to snapshot.
        Uses 'git ls-files' if work_dir is a git repository, otherwise includes all files.
        """
        if os.path.exists(os.path.join(work_dir, '.git')):
            try:
                # Use git ls-files to get tracked files
                result = subprocess.run(
                    ["git", "-C", str(work_dir), "ls-files"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return [os.path.join(work_dir, f) for f in result.stdout.splitlines()]
            except Exception:
                pass # Fallback to all files if git command fails

        # If not a git repo or git command failed, include all files
        tracked_files = []
        for root, _, files in os.walk(work_dir):
            for file in files:
                tracked_files.append(os.path.join(root, file))
        return tracked_files


    def create_snapshot(self, work_dir, label=''):
        """
        Creates a new snapshot of the specified working directory.

        Args:
            work_dir (str): The path to the working directory to snapshot.
            label (str): An optional label for the snapshot.

        Returns:
            str: The timestamp of the created snapshot.
        """
        with self._lock:
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            snapshot_dir = self._get_snapshot_path(timestamp)
            os.makedirs(snapshot_dir, exist_ok=True)

            tracked_files = self._get_tracked_files(work_dir)

            for src_path in tracked_files:
                relative_path = os.path.relpath(src_path, work_dir)
                dest_path = os.path.join(snapshot_dir, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)

            # Store metadata
            with open(os.path.join(snapshot_dir, 'metadata.txt'), 'w') as f:
                f.write(f"label: {label}\n")
                f.write(f"work_dir: {work_dir}\n")

            self._cleanup_old_snapshots()
            return timestamp

    def list_snapshots(self):
        """
        Lists all available snapshots.

        Returns:
            list: A list of dictionaries, each representing a snapshot with
                  'id' (timestamp), 'label', and 'work_dir'.
        """
        with self._lock:
            snapshots = []
            for snapshot_id in sorted(os.listdir(self.base_dir), reverse=True):
                snapshot_dir = self._get_snapshot_path(snapshot_id)
                if os.path.isdir(snapshot_dir):
                    label = "N/A"
                    original_work_dir = "N/A"
                    metadata_path = os.path.join(snapshot_dir, 'metadata.txt')
                    if os.path.exists(metadata_path):
                        with open(metadata_path) as f:
                            for line in f:
                                if line.startswith("label:"):
                                    label = line.split(":", 1)[1].strip()
                                elif line.startswith("work_dir:"):
                                    original_work_dir = line.split(":", 1)[1].strip()
                    snapshots.append({
                        'id': snapshot_id,
                        'label': label,
                        'work_dir': original_work_dir
                    })
            return snapshots

    def restore_snapshot(self, snapshot_id):
        """
        Restores a snapshot to its original working directory.

        Args:
            snapshot_id (str): The ID (timestamp) of the snapshot to restore.

        Raises:
            ValueError: If the snapshot_id is not found or work_dir is not in metadata.
        """
        with self._lock:
            snapshot_dir = self._get_snapshot_path(snapshot_id)
            if not os.path.isdir(snapshot_dir):
                raise ValueError(f"Snapshot '{snapshot_id}' not found.")

            original_work_dir = None
            metadata_path = os.path.join(snapshot_dir, 'metadata.txt')
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    for line in f:
                        if line.startswith("work_dir:"):
                            original_work_dir = line.split(":", 1)[1].strip()
                            break

            if not original_work_dir:
                raise ValueError(f"Original work_dir not found in metadata for snapshot '{snapshot_id}'.")

            # Clear current work_dir content (except .git if it exists and is tracked)
            if os.path.exists(original_work_dir):
                for item in os.listdir(original_work_dir):
                    if item in ('.git', '.hg', '.svn'):
                        continue
                    item_path = os.path.join(original_work_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

            # Copy snapshot files back
            for item in os.listdir(snapshot_dir):
                if item == 'metadata.txt':
                    continue
                src_path = os.path.join(snapshot_dir, item)
                dest_path = os.path.join(original_work_dir, item)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dest_path)
                else:
                    shutil.copy2(src_path, dest_path)

    def ensure_checkpoint(self, work_dir, label, recent_threshold_seconds=300):
        """
        Ensures a checkpoint exists for a given label, creating one only if
        no recent checkpoint with that label exists.

        Args:
            work_dir (str): The path to the working directory.
            label (str): The label for the checkpoint.
            recent_threshold_seconds (int): Time in seconds to consider a checkpoint "recent".
                                            Defaults to 300 seconds (5 minutes).

        Returns:
            str: The ID of the existing or newly created snapshot.
        """
        with self._lock:
            snapshots = self.list_snapshots()
            for snapshot in snapshots:
                if snapshot['label'] == label and snapshot['work_dir'] == work_dir:
                    snapshot_time = datetime.datetime.strptime(snapshot['id'], '%Y%m%d%H%M%S')
                    if (datetime.datetime.now() - snapshot_time).total_seconds() < recent_threshold_seconds:
                        print(f"Recent snapshot for label '{label}' found: {snapshot['id']}. Skipping new checkpoint.")
                        return snapshot['id']

            print(f"No recent snapshot for label '{label}' found. Creating new checkpoint.")
            return self.create_snapshot(work_dir, label)

    def _cleanup_old_snapshots(self):
        """
        Removes oldest snapshots if the total number exceeds max_snapshots.
        """
        snapshots = sorted(os.listdir(self.base_dir))
        while len(snapshots) > self.max_snapshots:
            oldest_snapshot_id = snapshots.pop(0)
            oldest_snapshot_dir = self._get_snapshot_path(oldest_snapshot_id)
            if os.path.isdir(oldest_snapshot_dir):
                shutil.rmtree(oldest_snapshot_dir)
                print(f"Cleaned up old snapshot: {oldest_snapshot_id}")

