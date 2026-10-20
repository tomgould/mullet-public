import shutil
import unittest
from pathlib import Path

from mullet.core.tools.file_ops import batch_copy, batch_move, batch_rename


class TestFileOperations(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("test_temp_dir")
        self.test_dir.mkdir(exist_ok=True)
        self.src_dir = self.test_dir / "src"
        self.dest_dir = self.test_dir / "dest"
        self.src_dir.mkdir(exist_ok=True)
        self.dest_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_batch_rename_basic(self):
        (self.src_dir / "file_1.txt").touch()
        (self.src_dir / "file_2.txt").touch()
        renamed = batch_rename(str(self.src_dir), "file", "document")
        self.assertEqual(len(renamed), 2)
        self.assertTrue((self.src_dir / "document_1.txt").exists())
        self.assertTrue((self.src_dir / "document_2.txt").exists())

    def test_batch_rename_no_match(self):
        (self.src_dir / "image_1.png").touch()
        renamed = batch_rename(str(self.src_dir), "file", "document")
        self.assertEqual(len(renamed), 0)
        self.assertTrue((self.src_dir / "image_1.png").exists())

    def test_batch_rename_empty_dir(self):
        renamed = batch_rename(str(self.src_dir), "file", "document")
        self.assertEqual(len(renamed), 0)

    def test_batch_copy_basic(self):
        (self.src_dir / "a.txt").touch()
        (self.src_dir / "b.txt").touch()
        copied = batch_copy(str(self.src_dir / "*.txt"), str(self.dest_dir))
        self.assertEqual(len(copied), 2)
        self.assertTrue((self.dest_dir / "a.txt").exists())
        self.assertTrue((self.dest_dir / "b.txt").exists())
        self.assertTrue((self.src_dir / "a.txt").exists())  # Original should still exist

    def test_batch_copy_no_match(self):
        (self.src_dir / "a.png").touch()
        copied = batch_copy(str(self.src_dir / "*.txt"), str(self.dest_dir))
        self.assertEqual(len(copied), 0)
        self.assertTrue((self.src_dir / "a.png").exists())
        self.assertFalse((self.dest_dir / "a.png").exists())

    def test_batch_copy_empty_src_dir(self):
        copied = batch_copy(str(self.src_dir / "*.txt"), str(self.dest_dir))
        self.assertEqual(len(copied), 0)

    def test_batch_move_basic(self):
        (self.src_dir / "x.doc").touch()
        (self.src_dir / "y.doc").touch()
        moved = batch_move(str(self.src_dir / "*.doc"), str(self.dest_dir))
        self.assertEqual(len(moved), 2)
        self.assertTrue((self.dest_dir / "x.doc").exists())
        self.assertTrue((self.dest_dir / "y.doc").exists())
        self.assertFalse((self.src_dir / "x.doc").exists())  # Original should be gone

    def test_batch_move_no_match(self):
        (self.src_dir / "x.jpg").touch()
        moved = batch_move(str(self.src_dir / "*.doc"), str(self.dest_dir))
        self.assertEqual(len(moved), 0)
        self.assertTrue((self.src_dir / "x.jpg").exists())
        self.assertFalse((self.dest_dir / "x.jpg").exists())

    def test_batch_move_empty_src_dir(self):
        moved = batch_move(str(self.src_dir / "*.doc"), str(self.dest_dir))
        self.assertEqual(len(moved), 0)

    def test_batch_rename_with_subdirectories(self):
        sub_dir = self.src_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "file_in_sub.txt").touch()
        (self.src_dir / "file_top.txt").touch()
        renamed = batch_rename(str(self.src_dir), "file", "new_file")
        self.assertEqual(len(renamed), 1) # Only top level file should be renamed
        self.assertTrue((self.src_dir / "new_file_top.txt").exists())
        self.assertTrue((sub_dir / "file_in_sub.txt").exists())

