import pytest
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from voc import VOCProcessor  # Your main class file


class TestVOCProcessor:
    @pytest.fixture
    def setup_test_environment(self):
        """Create a temporary test directory with sample files"""
        # Create test directories
        test_dir = Path("./test_data")
        train_dir = test_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        # Create test files with specific scenarios
        self.create_test_cases(train_dir)

        yield test_dir

        # Cleanup after tests
        shutil.rmtree(test_dir)
        if Path("./unwanted").exists():
            shutil.rmtree("./unwanted")
        if Path("./no_name").exists():
            shutil.rmtree("./no_name")

    def create_test_cases(self, train_dir):
        """Create sample files for different test scenarios"""
        # Case 1: Valid file (has both Person and Ladder)
        self.create_xml_file(
            train_dir / "valid.xml", labels=["Person", "Ladder"], valid_bbox=True
        )
        self.create_jpg_file(train_dir / "valid.jpg")

        # Case 2: Missing Person
        self.create_xml_file(
            train_dir / "missing_person.xml", labels=["Ladder"], valid_bbox=True
        )
        self.create_jpg_file(train_dir / "missing_person.jpg")

        # Case 3: Missing Ladder
        self.create_xml_file(
            train_dir / "missing_ladder.xml", labels=["Person"], valid_bbox=True
        )
        self.create_jpg_file(train_dir / "missing_ladder.jpg")

        # Case 4: Invalid bbox
        self.create_xml_file(
            train_dir / "invalid_bbox.xml",
            labels=["Person", "Ladder"],
            valid_bbox=False,
        )
        self.create_jpg_file(train_dir / "invalid_bbox.jpg")

        # Case 5: Unpaired file
        self.create_xml_file(
            train_dir / "unpaired.xml", labels=["Person", "Ladder"], valid_bbox=True
        )

        # Case 6: empty file
        self.create_xml_file(
            train_dir / "empty.xml",
            labels=[],  # No labels/objects
            valid_bbox=True,
        )
        self.create_jpg_file(train_dir / "empty.jpg")

    def create_xml_file(self, path: Path, labels: list, valid_bbox: bool):
        """Create a test XML file with specified labels and bbox validity"""
        root = ET.Element("annotation")

        # Add size element
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "640"
        ET.SubElement(size, "height").text = "640"

        # Add filename
        ET.SubElement(root, "filename").text = path.stem + ".jpg"

        # Add objects
        for label in labels:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = label
            bbox = ET.SubElement(obj, "bndbox")
            if valid_bbox:
                ET.SubElement(bbox, "xmin").text = "100"
                ET.SubElement(bbox, "ymin").text = "100"
                ET.SubElement(bbox, "xmax").text = "200"
                ET.SubElement(bbox, "ymax").text = "200"
            else:
                ET.SubElement(bbox, "xmin").text = "-10"  # Invalid coordinate
                ET.SubElement(bbox, "ymin").text = "100"
                ET.SubElement(bbox, "xmax").text = "200"
                ET.SubElement(bbox, "ymax").text = "200"

        tree = ET.ElementTree(root)
        tree.write(path)

    def create_jpg_file(self, path: Path):
        """Create an empty JPG file"""
        path.touch()

    def test_xml_processing(self, setup_test_environment):
        processor = VOCProcessor(str(setup_test_environment / "train"))

        # Test valid XML with both Person and Ladder
        xml_path = setup_test_environment / "train" / "valid.xml"
        is_valid, annotations = processor.process_xml(xml_path)
        assert is_valid == True
        assert len(annotations) == 2  # Should have 2 annotations (Person and Ladder)

        # Test invalid XML with only Person
        xml_path = setup_test_environment / "train" / "invalid_person.xml"
        is_valid, annotations = processor.process_xml(xml_path)
        assert is_valid == False
        assert len(annotations) == 0

    def test_file_movement(self, setup_test_environment):
        """Test that files are moved to the correct directories"""
        processor = VOCProcessor(str(setup_test_environment / "train"))
        processor.process_files()

        # Check processed directory (valid files)
        processed_files = list(Path("./no_name/train").glob("*"))
        assert (
            len(processed_files) == 1
        ), "Should have only valid.jpg in processed directory"

        # Define expected structure
        expected_structure = {
            "valid": {
                "directory": "./no_name/train",
                "files": {"valid.jpg"},
            },
            "unwanted": {
                "missing_person": {
                    "directory": "./unwanted/missing_person",
                    "files": {"missing_person.xml", "missing_person.jpg"},
                },
                "missing_ladder": {
                    "directory": "./unwanted/missing_ladder",
                    "files": {"missing_ladder.xml", "missing_ladder.jpg"},
                },
                "unpaired_file": {
                    "directory": "./unwanted/unpaired_file",
                    "files": {"unpaired.xml"},
                },
                "empty_annotation": {
                    "directory": "./unwanted/empty_annotation",
                    "files": {"empty.xml", "empty.jpg"},
                },
            },
        }

        # Verify each directory and its contents
        for category, details in expected_structure["unwanted"].items():
            dir_path = Path(details["directory"])
            if not dir_path.exists():
                pytest.fail(f"Expected directory not found: {dir_path}")

            actual_files = {f.name for f in dir_path.glob("*")}
            expected_files = details["files"]

            assert actual_files == expected_files, (
                f"Mismatch in {category} directory.\n"
                f"Expected: {expected_files}\n"
                f"Found: {actual_files}"
            )

        # Verify processed directory
        processed_path = Path(expected_structure["valid"]["directory"])
        processed_files = {f.name for f in processed_path.glob("*")}
        assert (
            processed_files == expected_structure["valid"]["files"]
        ), "Mismatch in processed directory"

        # Optional: Print detailed file lists for debugging
        print("\nDetailed Directory Structure:")
        print("\nProcessed directory:")
        print(f"./no_name/train/: {list(Path('./no_name/train').glob('*'))}")

        print("\nUnwanted directories:")
        for category in expected_structure["unwanted"].keys():
            path = Path(f"./unwanted/{category}")
            if path.exists():
                print(f"{path}/: {list(path.glob('*'))}")

    def test_csv_generation(self, setup_test_environment):
        processor = VOCProcessor(str(setup_test_environment / "train"))
        processor.process_files()
        processor.create_csv("test_annotations.csv")

        # Check CSV contents
        import pandas as pd

        df = pd.read_csv("test_annotations.csv", header=None)
        assert len(df) == 2  # Should have 2 rows (Person and Ladder annotations)

        # Cleanup
        os.remove("test_annotations.csv")

    def test_coordinate_normalization(self, setup_test_environment):
        processor = VOCProcessor(str(setup_test_environment / "train"))

        xml_path = setup_test_environment / "train" / "valid.xml"
        is_valid, annotations = processor.process_xml(xml_path)

        # Check normalized coordinates
        for annotation in annotations:
            assert 0 <= annotation[2] <= 1  # x_min
            assert 0 <= annotation[3] <= 1  # y_min
            assert 0 <= annotation[6] <= 1  # x_max
            assert 0 <= annotation[7] <= 1  # y_max

    def test_reason_categorization(self, setup_test_environment):
        """Test that files are properly categorized by reason"""
        processor = VOCProcessor(str(setup_test_environment / "train"))
        processor.process_files()

        # Check each unwanted directory
        assert len(list(Path("./unwanted/missing_person").glob("*"))) > 0
        assert len(list(Path("./unwanted/missing_ladder").glob("*"))) > 0

        # Verify log file exists and contains entries
        log_file = Path("./unwanted/unwanted_reasons.log")
        assert log_file.exists()

        # Print summary for verification
        processor.print_summary()

    def test_file_categorization(self, setup_test_environment):
        """Test that files are correctly categorized by reason"""
        processor = VOCProcessor(str(setup_test_environment / "train"))
        processor.process_files()

        # Check processed directory (valid files)
        processed_files = set(f.name for f in Path("./no_name/train").glob("*"))
        assert processed_files == {"valid.jpg"}

        # Check each unwanted category
        unwanted_categories = {
            "missing_person": {"missing_person.xml", "missing_person.jpg"},
            "missing_ladder": {"missing_ladder.xml", "missing_ladder.jpg"},
            "invalid_bbox": {"invalid_bbox.xml", "invalid_bbox.jpg"},
            "unpaired_file": {"unpaired.xml"},
        }

        for category, expected_files in unwanted_categories.items():
            category_path = Path("./unwanted") / category
            actual_files = set(f.name for f in category_path.glob("*"))
            assert actual_files == expected_files, f"Mismatch in {category} directory"

        # Verify log file exists and contains entries
        log_file = Path("./unwanted/unwanted_reasons.log")
        assert log_file.exists()

        # Print summary for verification
        processor.print_summary()


if __name__ == "__main__":
    pytest.main([__file__])
