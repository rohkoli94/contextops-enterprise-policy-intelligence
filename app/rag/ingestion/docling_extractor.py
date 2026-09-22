import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from app.domain.document import Document
from app.domain.document_element import (
    ContentType,
    DocumentElement,
)
from app.prompts.vision import (
    VISION_ANALYSIS_SYSTEM_PROMPT,
    VISION_ANALYSIS_USER_PROMPT,
)
from app.providers.llm.base import (
    LLMProvider,
    VisionRequest,
)
from app.rag.ingestion.base import DocumentExtractor
from app.utils.hashing import calculate_content_hash


class DoclingDocumentExtractor(DocumentExtractor):
    """
    Extracts structured document elements using Docling.

    Supported document elements:
    - TEXT
    - TABLE
    - IMAGE

    Visual processing flow:
    IMAGE
        ->
    Vision-capable LLM
        ->
    IMAGE / CHART / DIAGRAM
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
    ) -> None:
        # Create the Docling converter once and reuse it.
        self.converter = DocumentConverter()

        # IMPORTANT:
        # Docling can detect PictureItem objects without generating
        # their actual image payload.
        #
        # ContextOps needs the actual image so that it can be passed
        # to the vision-capable LLM.
        pdf_format_option = self.converter.format_to_options[
            InputFormat.PDF
        ]

        pdf_format_option.pipeline_options.generate_picture_images = True

        # The LLM provider is responsible for communicating
        # with the configured vision-capable model.
        self.llm_provider = llm_provider

    def extract(
        self,
        document: Document,
        stream: BinaryIO,
        file_name: str,
        document_version_id: uuid.UUID,
    ) -> list[DocumentElement]:
        """
        Main document extraction flow.

        1. Copy the incoming stream to a temporary file.
        2. Parse the document using Docling.
        3. Iterate through extracted structured items.
        4. Resolve each item's element type.
        5. Extract text, tables, or visuals.
        6. For visuals, call the vision model for semantic analysis.
        """

        temp_file_path: Path | None = None

        try:
            # Create a temporary file without loading the complete
            # source document into RAM.
            temp_file_path = self._create_temp_file(
                stream=stream,
                file_name=file_name,
            )

            # Parse the document using Docling.
            result = self.converter.convert(temp_file_path)

            elements: list[DocumentElement] = []

            # Iterate through all structured items extracted by Docling.
            #
            # item  -> actual extracted document content/item
            # level -> hierarchy depth of the item in the document
            for item, level in result.document.iterate_items():

                # Determine what type of domain DocumentElement
                # should be created from the Docling item.
                element_type = self._resolve_element_type(item)

                if element_type is None:
                    continue

                element: DocumentElement | None = None

                match element_type:

                    case ContentType.TEXT:
                        element = self._extract_text_element(
                            item=item,
                            level=level,
                            document=document,
                            file_name=file_name,
                        )

                    case ContentType.TABLE:
                        element = self._extract_table_element(
                            item=item,
                            level=level,
                            document=document,
                            file_name=file_name,
                        )

                    case ContentType.IMAGE:
                        element = self._extract_image_element(
                            item=item,
                            level=level,
                            document=document,
                            docling_document=result.document,
                            file_name=file_name,
                        )

                # Add only successfully extracted elements.
                if element is not None:
                    element.document_version_id = document_version_id
                    elements.append(element)

            return elements

        finally:
            # Always delete the temporary file, even when document
            # parsing or extraction fails.
            if temp_file_path is not None:
                temp_file_path.unlink(missing_ok=True)

    def _create_temp_file(
        self,
        stream: BinaryIO,
        file_name: str,
    ) -> Path:
        """
        Copy the incoming document stream to a temporary file.

        A 1 MB buffer is used so the complete document is not loaded
        into application memory at once.
        """

        # Preserve the original extension so Docling can identify
        # the document format correctly.
        suffix = Path(file_name).suffix or ".tmp"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:

            # Copy the stream incrementally.
            shutil.copyfileobj(
                stream,
                temp_file,
                length=1024 * 1024,
            )

            return Path(temp_file.name)

    def _resolve_element_type(
        self,
        item: Any,
    ) -> ContentType | None:
        """
        Determine which DocumentElement type should be created
        from the Docling item.

        Note:
        This is NOT the file content type or MIME type.
        It determines the type of our domain DocumentElement.
        """

        label = getattr(item, "label", None)

        if label is None:
            return None

        # Docling label may be an Enum or a normal string.
        label_value = (
            label.value
            if hasattr(label, "value")
            else str(label)
        ).lower()

        match label_value:

            # Text-based document elements.
            case (
                "text"
                | "paragraph"
                | "title"
                | "section_header"
            ):
                return ContentType.TEXT

            # Structured table.
            case "table":
                return ContentType.TABLE

            # Visual content.
            #
            # At this stage Docling tells us it is a picture/visual.
            # The vision model later determines whether it is:
            # IMAGE / CHART / DIAGRAM.
            case "picture" | "image":
                return ContentType.IMAGE

            case _:
                return None

    def _extract_text_element(
        self,
        item: Any,
        level: int,
        document: Document,
        file_name: str,
    ) -> DocumentElement | None:
        """
        Extract a text item and convert it into a TEXT
        DocumentElement.
        """

        content = getattr(item, "text", "")

        if not isinstance(content, str):
            return None

        content = content.strip()

        # Skip empty content.
        if not content:
            return None

        return DocumentElement(
            element_id=str(uuid.uuid4()),
            document_id=document.document_id,
            page_number=self._get_page_number(item),
            content_type=ContentType.TEXT,
            content=content,
            content_hash=calculate_content_hash(content),
            metadata=self._extract_metadata(
                item=item,
                level=level,
                file_name=file_name,
            ),
        )

    def _extract_table_element(
        self,
        item: Any,
        level: int,
        document: Document,
        file_name: str,
    ) -> DocumentElement | None:
        """
        Extract a Docling TableItem and convert its TableData
        grid into Markdown.

        This implementation intentionally uses TableData.grid
        because the installed Docling version does not expose
        export_to_markdown().
        """

        data = getattr(item, "data", None)

        if data is None:
            return None

        grid = getattr(data, "grid", None)

        if not grid:
            return None

        rows: list[list[str]] = []

        for row in grid:
            cells: list[str] = []

            for cell in row:
                cell_text = getattr(cell, "text", "")

                if cell_text is None:
                    cell_text = ""

                # Escape Markdown pipe characters so cell contents
                # cannot break the table structure.
                cell_text = str(cell_text).strip()
                cell_text = cell_text.replace("|", "\\|")

                cells.append(cell_text)

            rows.append(cells)

        if not rows:
            return None

        # Determine the maximum number of columns.
        column_count = max(
            (len(row) for row in rows),
            default=0,
        )

        if column_count == 0:
            return None

        # Normalize rows to the same column count.
        for row in rows:
            while len(row) < column_count:
                row.append("")

        # First row becomes the Markdown header.
        header = rows[0]

        markdown_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(
                ["---"] * column_count
            ) + " |",
        ]

        # Remaining rows become table body.
        for row in rows[1:]:
            markdown_lines.append(
                "| " + " | ".join(row) + " |"
            )

        content = "\n".join(markdown_lines).strip()

        # Skip empty tables.
        if not content:
            return None

        metadata = self._extract_metadata(
            item=item,
            level=level,
            file_name=file_name,
        )

        # Indicates how the table content is represented.
        metadata["content_format"] = "markdown"

        # Preserve useful table dimensions.
        num_rows = getattr(data, "num_rows", None)
        num_cols = getattr(data, "num_cols", None)

        if num_rows is not None:
            metadata["table_rows"] = num_rows

        if num_cols is not None:
            metadata["table_columns"] = num_cols

        return DocumentElement(
            element_id=str(uuid.uuid4()),
            document_id=document.document_id,
            page_number=self._get_page_number(item),
            content_type=ContentType.TABLE,
            content=content,
            content_hash=calculate_content_hash(content),
            metadata=metadata,
        )

    def _extract_image_element(
        self,
        item: Any,
        level: int,
        document: Document,
        docling_document: Any,
        file_name: str,
    ) -> DocumentElement | None:
        """
        Extract a visual from Docling and analyze it using a
        vision-capable LLM.

        Flow:

        PictureItem
            ->
        item.get_image(docling_document)
            ->
        PIL Image
            ->
        image bytes + media type
            ->
        VisionRequest
            ->
        Vision-capable LLM
            ->
        IMAGE / CHART / DIAGRAM + description
        """

        # Extract the actual visual and determine its media type.
        image_data = self._extract_image_data(
            item=item,
            docling_document=docling_document,
        )

        if image_data is None:
            return None

        image_bytes, media_type = image_data

        # Send the extracted visual to the vision-capable LLM.
        vision_response = self.llm_provider.generate_vision(
            VisionRequest(
                system_prompt=VISION_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=VISION_ANALYSIS_USER_PROMPT,
                image_bytes=image_bytes,
                media_type=media_type,
            )
        )

        # Parse the model response and determine whether the
        # visual is an IMAGE, CHART, or DIAGRAM.
        visual_type, description = self._parse_vision_response(
            vision_response.content
        )

        metadata = self._extract_metadata(
            item=item,
            level=level,
            file_name=file_name,
        )

        # Indicates that this visual has been processed by
        # the vision-capable LLM.
        metadata["vision_analyzed"] = True

        # Store information about the model/provider used
        # for future tracing and debugging.
        metadata["vision_model"] = vision_response.model
        metadata["vision_provider"] = vision_response.provider

        # Store the detected visual type.
        metadata["visual_type"] = visual_type.value

        # Store the media type of the image sent to the vision model.
        metadata["media_type"] = media_type

        return DocumentElement(
            element_id=str(uuid.uuid4()),
            document_id=document.document_id,
            page_number=self._get_page_number(item),
            content_type=visual_type,
            content=description,
            content_hash=calculate_content_hash(description),
            metadata=metadata,
        )

    def _extract_image_data(
        self,
        item: Any,
        docling_document: Any,
    ) -> tuple[bytes, str] | None:
        """
        Extract the actual visual from a Docling PictureItem.

        Returns:
        - image bytes
        - dynamically determined media type

        Example:
        PNG  -> (bytes, "image/png")
        JPEG -> (bytes, "image/jpeg")
        """

        get_image = getattr(item, "get_image", None)

        if not callable(get_image):
            return None

        # Docling returns the extracted visual as a PIL Image.
        image = get_image(docling_document)

        if image is None:
            return None

        # PIL images do not always retain their original format.
        image_format = (
            getattr(image, "format", None) or "PNG"
        ).upper()

        # Map image formats to MIME/media types.
        media_types = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp",
        }

        # Fall back to PNG for unsupported or unknown formats.
        media_type = media_types.get(
            image_format,
            "image/png",
        )

        # If the format is not explicitly supported in the mapping,
        # save the image as PNG so the bytes and media type remain
        # consistent.
        output_format = (
            image_format
            if image_format in media_types
            else "PNG"
        )

        # Convert the PIL Image into bytes.
        image_buffer = BytesIO()

        image.save(
            image_buffer,
            format=output_format,
        )

        return (
            image_buffer.getvalue(),
            media_type,
        )

    def _parse_vision_response(
        self,
        content: str,
    ) -> tuple[ContentType, str]:
        """
        Parse the response returned by the vision model.

        Expected response format:

        TYPE: CHART
        DESCRIPTION: Bar chart showing revenue growth.
        """

        # Default to IMAGE if classification cannot be parsed.
        visual_type = ContentType.IMAGE

        # Default description is the complete model response.
        description = content.strip()

        lines = content.splitlines()

        for line in lines:
            normalized_line = line.strip()

            if normalized_line.upper().startswith("TYPE:"):

                type_value = (
                    normalized_line
                    .split(":", 1)[1]
                    .strip()
                    .upper()
                )

                match type_value:

                    case "CHART":
                        visual_type = ContentType.CHART

                    case "DIAGRAM":
                        visual_type = ContentType.DIAGRAM

                    case "IMAGE":
                        visual_type = ContentType.IMAGE

            elif normalized_line.upper().startswith(
                "DESCRIPTION:"
            ):
                description = (
                    normalized_line
                    .split(":", 1)[1]
                    .strip()
                )

        return visual_type, description

    def _get_page_number(
        self,
        item: Any,
    ) -> int:
        """
        Get the source page number from Docling provenance.

        Falls back to page 1 when provenance is unavailable.
        """

        provenance = getattr(item, "prov", None)

        if provenance:

            page_no = getattr(
                provenance[0],
                "page_no",
                None,
            )

            if page_no is not None:
                return page_no

        return 1

    def _extract_metadata(
        self,
        item: Any,
        level: int,
        file_name: str,
    ) -> dict[str, Any]:
        """
        Extract useful element-level metadata from Docling.

        Metadata is added only when available.
        """

        metadata: dict[str, Any] = {
            "file_name": file_name,
            "hierarchy_level": level,
        }

        # Document/layout label describing the original type
        # detected by Docling.
        label = getattr(item, "label", None)

        if label is not None:
            metadata["label"] = (
                label.value
                if hasattr(label, "value")
                else str(label)
            )

        # Provenance tells us where the extracted content
        # originally came from in the source document.
        provenance = []

        for prov in getattr(item, "prov", []) or []:

            prov_metadata: dict[str, Any] = {}

            # Original source page.
            page_no = getattr(
                prov,
                "page_no",
                None,
            )

            if page_no is not None:
                prov_metadata["page_number"] = page_no

            # Character range of the content in the source.
            charspan = getattr(
                prov,
                "charspan",
                None,
            )

            if charspan is not None:
                prov_metadata["charspan"] = list(charspan)

            # Physical location of the element on the page.
            bbox = getattr(
                prov,
                "bbox",
                None,
            )

            if bbox is not None:
                prov_metadata["bbox"] = {
                    "left": getattr(bbox, "l", None),
                    "top": getattr(bbox, "t", None),
                    "right": getattr(bbox, "r", None),
                    "bottom": getattr(bbox, "b", None),
                }

            if prov_metadata:
                provenance.append(prov_metadata)

        if provenance:
            metadata["provenance"] = provenance

        return metadata