import os
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


class UnwantedReason(Enum):
    MISSING_PERSON = "missing_person"
    MISSING_LADDER = "missing_ladder"
    UNPAIRED_FILE = "unpaired_file"
    EMPTY_ANNOTATION = "empty_annotation"
    INVALID_RESOLUTION = "invalid_resolution"
    CORRUPTED_FILE = "corrupted_file"
    INVALID_BBOX = "invalid_bbox"


class VOCProcessor:
    def __init__(self, voc_filepath: str, annotation_limit: int = 300):
        self.voc_path = Path(voc_filepath)
        self.unwanted_base = Path("./unwanted")
        self.processed_dir = Path("./no_name/train")
        self.annotations: List[List] = []
        self.annotation_limit = annotation_limit
        self.annotation_count = 0  # Track total annotations

        # Create necessary directories for each reason
        self.unwanted_dirs = {}
        for reason in UnwantedReason:
            path = self.unwanted_base / reason.value
            path.mkdir(parents=True, exist_ok=True)
            self.unwanted_dirs[reason] = path

        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_xml(self, xml_path: Path) -> Tuple[bool, List]:
        """
        Process a single XML file and return annotations if valid.
        Valid files must contain BOTH 'Ladder' AND 'Person' labels.
        Returns: (is_valid, annotations_list)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get all object labels in the XML
            labels = set(obj.find("name").text for obj in root.findall("object"))

            # Check if both required labels are present
            if not {"Ladder", "Person"}.issubset(labels):
                return False, []

            filename = root.find("filename").text
            width = float(root.find("size/width").text)
            height = float(root.find("size/height").text)

            file_annotations = []
            image_path = f"gs://i220342h-3386-aip/train/{filename}"

            # Process all objects in the XML
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")

                # Normalize coordinates with single division operation
                x_min = round(float(bbox.find("xmin").text) / width, 9)
                y_min = round(float(bbox.find("ymin").text) / height, 9)
                x_max = round(float(bbox.find("xmax").text) / width, 9)
                y_max = round(float(bbox.find("ymax").text) / height, 9)

                file_annotations.append(
                    [
                        image_path,
                        obj.find("name").text,
                        x_min,
                        y_min,
                        "",
                        "",
                        x_max,
                        y_max,
                        "",
                        "",
                    ]
                )

            return True, file_annotations

        except Exception as e:
            print(f"Error processing {xml_path}: {str(e)}")
            return False, []

    def verify_file_pairs(self, xml_path: Path) -> bool:
        """
        Verify that an XML file has a matching JPG file.
        Returns True if the pair exists, False otherwise.
        """
        # Get the corresponding jpg path by replacing .xml with .jpg
        jpg_path = xml_path.with_suffix(".jpg")

        # Check if jpg exists
        if not jpg_path.exists():
            print(f"Warning: Missing JPG file for {xml_path.name}")
            return False

        # Optional: Check if files are readable
        try:
            # Try to open both files to ensure they're not corrupted
            with open(xml_path, "rb") as xml_file:
                xml_file.read(1024)  # Read first 1KB to check if readable
            with open(jpg_path, "rb") as jpg_file:
                jpg_file.read(1024)  # Read first 1KB to check if readable
            return True
        except IOError as e:
            print(f"Error reading files for {xml_path.name}: {str(e)}")
            return False

    def process_files(self):
        """Process all files with reason checking"""
        xml_files = list(self.voc_path.glob("*.xml"))

        print(f"Processing files (limit: {self.annotation_limit} annotations)...")

        for xml_path in xml_files:
            # Check for issues
            issues = self.check_file_issues(xml_path)
            if self.annotation_count >= self.annotation_limit:
                print(f"\nReached annotation limit ({self.annotation_limit})")
                break

            if issues:
                # Move to unwanted with reasons
                self.move_to_unwanted(xml_path, issues)
            else:
                # Process valid file
                is_valid, annotations = self.process_xml(xml_path)
                if is_valid:
                    new_total = self.annotation_count + len(annotations)
                    if new_total > self.annotation_limit:
                        print(
                            f"\nSkipping {xml_path.name} as it would exceed annotation limit"
                        )
                        continue
                    self.annotations.extend(annotations)
                    self.annotation_count = new_total
                    # Move to processed directory
                    jpg_path = xml_path.with_suffix(".jpg")
                    shutil.move(str(jpg_path), str(self.processed_dir / jpg_path.name))
                    shutil.move(str(xml_path), str(self.processed_dir / xml_path.name))

    def create_csv(self, output_path: str = "annotations.csv"):
        """Create the final CSV file"""
        if not self.annotations:
            print("No valid annotations found!")
            return

        df = pd.DataFrame(
            self.annotations,
            columns=[
                "image_file_name",
                "label",
                "X_MIN",
                "Y_MIN",
                "",
                "",
                "X_MAX",
                "Y_MAX",
                "",
                "",
            ],
        )
        df.to_csv(output_path, index=False, header=False)

    def print_summary(self):
        """Print detailed processing summary with annotation limit info"""
        print("\nProcessing Summary:")
        print(f"Annotation limit: {self.annotation_limit}")
        print(f"Total annotations collected: {self.annotation_count}")
        print(f"Processed images: {len(list(self.processed_dir.glob('*.jpg')))}")

        # Count files in each unwanted directory
        print("\nUnwanted files by reason:")
        total_unwanted = 0
        for reason in UnwantedReason:
            count = len(list(self.unwanted_dirs[reason].glob("*")))
            total_unwanted += count
            print(f"{reason.value}: {count} files")

        # Print progress towards limit
        print(
            f"\nAnnotation Progress: {self.annotation_count}/{self.annotation_limit} "
            f"({(self.annotation_count/self.annotation_limit)*100:.1f}%)"
        )

    def check_file_issues(self, xml_path: Path) -> List[Tuple[UnwantedReason, str]]:
        """
        Check for various issues in the XML and corresponding JPG file.
        Returns: List of tuples (UnwantedReason, detailed_message)
        """
        issues = []

        try:
            # Check for paired jpg
            jpg_path = xml_path.with_suffix(".jpg")
            if not jpg_path.exists():
                issues.append(
                    (UnwantedReason.UNPAIRED_FILE, f"Missing JPG file: {jpg_path.name}")
                )
                return issues

            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Check for empty annotations (no objects)
            objects = root.findall("object")
            if not objects:
                issues.append(
                    (UnwantedReason.EMPTY_ANNOTATION, "No object tags found in XML")
                )
                return issues

            # Check for required labels
            labels = set(obj.find("name").text for obj in objects)
            if "Person" not in labels:
                issues.append(
                    (
                        UnwantedReason.MISSING_PERSON,
                        f"Missing 'Person' label. Found labels: {labels}",
                    )
                )
            if "Ladder" not in labels:
                issues.append(
                    (
                        UnwantedReason.MISSING_LADDER,
                        f"Missing 'Ladder' label. Found labels: {labels}",
                    )
                )

            # Check image resolution
            size_elem = root.find("size")
            if size_elem is not None:
                width = float(size_elem.find("width").text)
                height = float(size_elem.find("height").text)
                if width != 640 or height != 640:
                    issues.append(
                        (
                            UnwantedReason.INVALID_RESOLUTION,
                            f"Invalid resolution: {width}x{height}, expected 640x640",
                        )
                    )

            # Check and correct bounding box validity
            self.correct_bbox_coordinates(xml_path)

            # Recheck after corrections
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size_elem = root.find("size")
            width = float(size_elem.find("width").text)
            height = float(size_elem.find("height").text)

            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                if bbox is not None:
                    is_valid, error_msg, _, _ = self.check_bbox_validity(
                        bbox, width, height
                    )
                    if not is_valid:
                        label = obj.find("name").text
                        detailed_msg = f"Invalid bbox for {label}: {error_msg}"
                        issues.append((UnwantedReason.INVALID_BBOX, detailed_msg))
                        break

        except ET.ParseError as e:
            issues.append(
                (UnwantedReason.CORRUPTED_FILE, f"XML parsing error: {str(e)}")
            )
        except Exception as e:
            issues.append(
                (UnwantedReason.CORRUPTED_FILE, f"Unexpected error: {str(e)}")
            )

        return issues

    def check_bbox_validity(
        self, bbox, width: float, height: float
    ) -> Tuple[bool, str, bool, dict]:
        """
        Check bbox coordinates validity and return detailed error message.
        Returns: (is_valid, error_message, needs_correction, corrected_values)
        """
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        needs_correction = False
        corrected_values = {}

        # Correct 641 to 640
        if xmax == 641:
            xmax = 640
            needs_correction = True
            corrected_values["xmax"] = 640

        if ymax == 641:
            ymax = 640
            needs_correction = True
            corrected_values["ymax"] = 640

        # Check each condition separately with detailed messages
        if xmin < 0:
            return False, f"xmin ({xmin}) < 0", False, {}
        if ymin < 0:
            return False, f"ymin ({ymin}) < 0", False, {}
        if xmax > width and xmax != 640:  # Allow 640
            return False, f"xmax ({xmax}) > width ({width})", False, {}
        if ymax > height and ymax != 640:  # Allow 640
            return False, f"ymax ({ymax}) > height ({height})", False, {}
        if xmin >= xmax:
            return False, f"xmin ({xmin}) >= xmax ({xmax})", False, {}
        if ymin >= ymax:
            return False, f"ymin ({ymin}) >= ymax ({ymax})", False, {}

        return True, "", needs_correction, corrected_values

    def correct_bbox_coordinates(self, xml_path: Path) -> bool:
        """
        Correct bbox coordinates in XML file if needed.
        Returns: True if corrections were made, False otherwise.
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            size_elem = root.find("size")
            width = float(size_elem.find("width").text)
            height = float(size_elem.find("height").text)

            corrections_made = False

            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                if bbox is not None:
                    is_valid, msg, needs_correction, corrected_values = (
                        self.check_bbox_validity(bbox, width, height)
                    )

                    if needs_correction:
                        corrections_made = True
                        for coord, value in corrected_values.items():
                            bbox.find(coord).text = str(int(value))

            if corrections_made:
                tree.write(xml_path)

            return corrections_made

        except Exception as e:
            print(f"Error correcting coordinates in {xml_path}: {str(e)}")
            return False

    def move_to_unwanted(
        self, file_path: Path, issues: List[Tuple[UnwantedReason, str]]
    ):
        """
        Modified to handle detailed error messages
        """
        if not issues:
            return

        # Use the first reason as the primary reason
        primary_reason, detailed_msg = issues[0]
        target_dir = self.unwanted_dirs[primary_reason]

        try:
            # Move files
            if file_path.suffix == ".xml":
                jpg_path = file_path.with_suffix(".jpg")
                if jpg_path.exists():
                    shutil.move(str(jpg_path), str(target_dir / jpg_path.name))
            shutil.move(str(file_path), str(target_dir / file_path.name))

            # Log with detailed messages
            with open(self.unwanted_base / "unwanted_reasons.log", "a") as f:
                f.write(f"{file_path.name}:\n")
                for reason, msg in issues:
                    f.write(f"  - {reason.value}: {msg}\n")
                f.write("\n")

        except Exception as e:
            print(f"Error moving {file_path}: {str(e)}")
