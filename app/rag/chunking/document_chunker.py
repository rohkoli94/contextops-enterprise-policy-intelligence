import re
import uuid
from dataclasses import dataclass, field

from app.domain.document_chunk import DocumentChunk
from app.domain.document_element import (
    ContentType,
    DocumentElement,
)
from app.rag.chunking.base import (
    DocumentChunker,
    TokenCounter,
)
from app.utils.hashing import calculate_content_hash


@dataclass
class _StructureGroup:
    """
    Logical document group created from the document hierarchy.

    Example:

        Remote Work Policy
            -> Eligibility
                -> Paragraph 1
                -> Paragraph 2
    """

    hierarchy_path: tuple[str, ...]
    elements: list[DocumentElement] = field(
        default_factory=list
    )


class HybridDocumentChunker(DocumentChunker):
    """
    Hybrid chunking strategy for ContextOps.

    Strategy:

        1. Preserve document hierarchy
        2. Use content-type-specific rules
        3. Create semantic candidate chunks
        4. Apply tokenizer-aware size constraints
        5. Produce final DocumentChunks

    Content-type strategies:

        TEXT
            Structure-aware + semantic grouping
            + tokenizer-aware refinement

        TABLE
            Preserve column names and row relationships
            + row-based splitting
            + tokenizer-aware refinement

        IMAGE / CHART / DIAGRAM
            Keep vision-generated description as a semantic unit
            + tokenizer-aware refinement if oversized
    """

    def __init__(
        self,
        max_tokens: int,
        token_counter: TokenCounter,
    ) -> None:
        """
        Args:
            max_tokens:
                Maximum number of tokens allowed in a final chunk.

            token_counter:
                Function that returns the token count for a string.

        The tokenizer is injected because the chunker should use
        the tokenizer associated with the embedding model rather
        than hard-code a specific tokenizer.
        """

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero"
            )

        self.max_tokens = max_tokens
        self.token_counter = token_counter

    def chunk(
        self,
        elements: list[DocumentElement],
    ) -> list[DocumentChunk]:
        """
        Convert extracted DocumentElements into final chunks.

        Overall flow:

            DocumentElement[]
                    ↓
            Structure-aware grouping
                    ↓
            Content-type routing
                    ↓
            Content-specific chunking
                    ↓
            Token-aware refinement
                    ↓
            DocumentChunk[]
        """

        if not elements:
            return []

        # --------------------------------------------------
        # STEP 1
        # Build logical groups from document structure.
        # --------------------------------------------------

        structure_groups = self._group_by_hierarchy(
            elements
        )

        chunks: list[DocumentChunk] = []

        # Chunk index is sequential across the complete
        # document version.
        next_chunk_index = 0

        # --------------------------------------------------
        # STEP 2
        # Route every structure group by content type.
        # --------------------------------------------------

        for group in structure_groups:
            content_groups = self._group_by_content_type(
                group.elements
            )

            for (
                content_type,
                content_elements,
            ) in content_groups:

                generated_chunks = (
                    self._chunk_by_content_type(
                        content_type=content_type,
                        elements=content_elements,
                        hierarchy_path=group.hierarchy_path,
                    )
                )

                # Assign final sequential chunk indexes.
                for chunk in generated_chunks:
                    chunk.chunk_index = next_chunk_index
                    next_chunk_index += 1
                    chunks.append(chunk)

        return chunks

    # =========================================================
    # STEP 1 — STRUCTURE-AWARE GROUPING
    # =========================================================

    def _group_by_hierarchy(
        self,
        elements: list[DocumentElement],
    ) -> list[_StructureGroup]:
        """
        Group elements according to the hierarchy extracted
        from the original document.

        Example:

            Level 0 -> Remote Work Policy
            Level 1 -> Eligibility
            Level 2 -> Existing Employees
            Level 3 -> Paragraph 1
            Level 3 -> Paragraph 2
            Level 1 -> Exceptions
            Level 2 -> Paragraph

        Logical result:

            Group 1
                path = ["Remote Work Policy"]
                elements = title

            Group 2
                path = ["Remote Work Policy", "Eligibility"]
                elements = Eligibility + related content

            Group 3
                path = ["Remote Work Policy",
                        "Eligibility",
                        "Existing Employees"]
                elements = Existing Employees + related content

            Group 4
                path = ["Remote Work Policy", "Exceptions"]
                elements = Exceptions + related content
        """

        groups: list[_StructureGroup] = []

        # Holds the active hierarchy.

        # Example:
        #
        # [
        #     (0, "Remote Work Policy"),
        #     (1, "Eligibility"),
        #     (2, "Existing Employees"),
        # ]
        hierarchy_stack: list[
            tuple[int, str]
        ] = []

        current_group: _StructureGroup | None = None

        for element in elements:
            level = self._get_hierarchy_level(
                element
            )

            # A structural element such as a title or heading
            # changes the current hierarchy.
            if self._is_structural_element(element):

                # Remove the current heading and anything below it.
                #
                # Example:
                #
                # Existing:
                # Level 0 -> Policy
                # Level 1 -> Eligibility
                # Level 2 -> Existing Employees
                #
                # New:
                # Level 1 -> Exceptions
                #
                # Result:
                # Level 0 -> Policy
                # Level 1 -> Exceptions
                while (
                    hierarchy_stack
                    and hierarchy_stack[-1][0] >= level
                ):
                    hierarchy_stack.pop()

                hierarchy_stack.append(
                    (
                        level,
                        element.content,
                    )
                )

                hierarchy_path = tuple(
                    content
                    for _, content in hierarchy_stack
                )

                current_group = _StructureGroup(
                    hierarchy_path=hierarchy_path
                )

                groups.append(current_group)

                # Keep the heading itself in its own
                # structural group.
                current_group.elements.append(
                    element
                )

                continue

            # Non-structural content belongs to the most recent
            # structural group.
            if current_group is None:
                current_group = _StructureGroup(
                    hierarchy_path=()
                )
                groups.append(current_group)

            current_group.elements.append(
                element
            )

        return groups

    # =========================================================
    # STEP 2 — CONTENT TYPE ROUTING
    # =========================================================

    def _group_by_content_type(
        self,
        elements: list[DocumentElement],
    ) -> list[
        tuple[
            ContentType,
            list[DocumentElement],
        ]
    ]:
        """
        Group consecutive elements by content type while
        preserving their original order.

        Example:

            TEXT
            TEXT
            TABLE
            TEXT
            CHART

        becomes:

            TEXT   -> [TEXT, TEXT]
            TABLE  -> [TABLE]
            TEXT   -> [TEXT]
            CHART  -> [CHART]
        """

        groups: list[
            tuple[
                ContentType,
                list[DocumentElement],
            ]
        ] = []

        current_type: ContentType | None = None
        current_elements: list[
            DocumentElement
        ] = []

        for element in elements:

            # Content type changed.
            if (
                current_type is not None
                and element.content_type != current_type
            ):
                groups.append(
                    (
                        current_type,
                        current_elements,
                    )
                )

                current_elements = []

            current_type = element.content_type
            current_elements.append(element)

        # Flush final group.
        if current_type is not None:
            groups.append(
                (
                    current_type,
                    current_elements,
                )
            )

        return groups

    # =========================================================
    # STEP 3 — CONTENT-TYPE STRATEGY
    # =========================================================

    def _chunk_by_content_type(
        self,
        content_type: ContentType,
        elements: list[DocumentElement],
        hierarchy_path: tuple[str, ...],
    ) -> list[DocumentChunk]:
        """
        Route content to its content-specific chunking strategy.
        """

        match content_type:

            case ContentType.TEXT:
                return self._chunk_text(
                    elements=elements,
                    hierarchy_path=hierarchy_path,
                )

            case ContentType.TABLE:
                return self._chunk_table(
                    elements=elements,
                    hierarchy_path=hierarchy_path,
                )

            case (
                ContentType.IMAGE
                | ContentType.CHART
                | ContentType.DIAGRAM
            ):
                return self._chunk_visual(
                    elements=elements,
                    hierarchy_path=hierarchy_path,
                )

            case _:
                return []

    # =========================================================
    # TEXT CHUNKING
    # =========================================================

    def _chunk_text(
        self,
        elements: list[DocumentElement],
        hierarchy_path: tuple[str, ...],
    ) -> list[DocumentChunk]:
        """
        Text strategy:

            Structure-aware grouping
                ↓
            Semantic candidate
                ↓
            Token check
                ↓
            Split only if needed
        """

        chunks: list[DocumentChunk] = []

        current_elements: list[
            DocumentElement
        ] = []

        current_content: list[str] = []

        for element in elements:

            # First handle an element that is already too large.
            if not self._fits_token_limit(
                element.content
            ):
                # Finalize any existing candidate first.
                if current_elements:
                    chunks.append(
                        self._build_chunk(
                            elements=current_elements,
                            content=self._join_content(
                                current_content
                            ),
                            hierarchy_path=hierarchy_path,
                        )
                    )

                    current_elements = []
                    current_content = []

                # Split this oversized element.
                chunks.extend(
                    self._split_oversized_text(
                        element=element,
                        hierarchy_path=hierarchy_path,
                    )
                )

                continue

            # Try adding the next element to the current
            # candidate chunk.
            candidate_content = self._join_content(
                current_content
                + [element.content]
            )

            if self._fits_token_limit(
                candidate_content
            ):
                current_elements.append(
                    element
                )
                current_content.append(
                    element.content
                )
                continue

            # Candidate exceeded token limit.
            #
            # Finalize the current candidate first.
            if current_elements:
                chunks.append(
                    self._build_chunk(
                        elements=current_elements,
                        content=self._join_content(
                            current_content
                        ),
                        hierarchy_path=hierarchy_path,
                    )
                )

            # Start a new candidate with the current element.
            current_elements = [element]
            current_content = [
                element.content
            ]

        # Flush remaining candidate.
        if current_elements:
            chunks.append(
                self._build_chunk(
                    elements=current_elements,
                    content=self._join_content(
                        current_content
                    ),
                    hierarchy_path=hierarchy_path,
                )
            )

        return chunks

    # =========================================================
    # TABLE CHUNKING
    # =========================================================

    def _chunk_table(
        self,
        elements: list[DocumentElement],
        hierarchy_path: tuple[str, ...],
    ) -> list[DocumentChunk]:
        """
        Table strategy:

            Small table
                -> Keep complete table together.

            Large table
                -> Split by rows.
                -> Repeat column names in each chunk.

            Oversized row
                -> Convert row into
                   "Column: Value" representation.
                -> Apply tokenizer-aware splitting.
        """

        chunks: list[DocumentChunk] = []

        for table_element in elements:

            lines = table_element.content.splitlines()

            # If the table does not look like a normal Markdown
            # table, treat it as a normal semantic unit.
            if len(lines) < 2:
                chunks.append(
                    self._build_chunk(
                        elements=[table_element],
                        content=table_element.content,
                        hierarchy_path=hierarchy_path,
                    )
                )
                continue

            # Markdown table:
            #
            # line 0 = column names
            # line 1 = separator
            # line 2+ = rows
            header = lines[0]
            separator = lines[1]
            rows = lines[2:]

            # --------------------------------------------------
            # Small table
            # --------------------------------------------------

            if self._fits_token_limit(
                table_element.content
            ):
                chunks.append(
                    self._build_chunk(
                        elements=[table_element],
                        content=table_element.content,
                        hierarchy_path=hierarchy_path,
                    )
                )
                continue

            # --------------------------------------------------
            # Large table
            # --------------------------------------------------

            current_rows: list[str] = []

            for row in rows:

                candidate_table = "\n".join(
                    [
                        header,
                        separator,
                        *current_rows,
                        row,
                    ]
                )

                # Still fits -> add this row.
                if self._fits_token_limit(
                    candidate_table
                ):
                    current_rows.append(row)
                    continue

                # Existing rows form a valid chunk.
                if current_rows:
                    chunks.append(
                        self._build_chunk(
                            elements=[table_element],
                            content="\n".join(
                                [
                                    header,
                                    separator,
                                    *current_rows,
                                ]
                            ),
                            hierarchy_path=hierarchy_path,
                        )
                    )

                # Check whether this row can fit by itself
                # with the column names.
                single_row_table = "\n".join(
                    [
                        header,
                        separator,
                        row,
                    ]
                )

                if self._fits_token_limit(
                    single_row_table
                ):
                    current_rows = [row]
                    continue

                # The row itself exceeds the token budget.
                chunks.extend(
                    self._split_oversized_table_row(
                        table_element=table_element,
                        header=header,
                        row=row,
                        hierarchy_path=hierarchy_path,
                    )
                )

                current_rows = []

            # Flush remaining rows.
            if current_rows:
                chunks.append(
                    self._build_chunk(
                        elements=[table_element],
                        content="\n".join(
                            [
                                header,
                                separator,
                                *current_rows,
                            ]
                        ),
                        hierarchy_path=hierarchy_path,
                    )
                )

        return chunks

    # =========================================================
    # VISUAL CHUNKING
    # =========================================================

    def _chunk_visual(
        self,
        elements: list[DocumentElement],
        hierarchy_path: tuple[str, ...],
    ) -> list[DocumentChunk]:
        """
        IMAGE / CHART / DIAGRAM strategy:

            Keep the vision-generated description as one
            semantic unit whenever it fits.

            If it exceeds the token limit, refine it using
            the oversized-text fallback.
        """

        chunks: list[DocumentChunk] = []

        for element in elements:

            if self._fits_token_limit(
                element.content
            ):
                chunks.append(
                    self._build_chunk(
                        elements=[element],
                        content=element.content,
                        hierarchy_path=hierarchy_path,
                    )
                )
                continue

            # Very large visual descriptions still need
            # tokenizer-aware refinement.
            chunks.extend(
                self._split_oversized_text(
                    element=element,
                    hierarchy_path=hierarchy_path,
                )
            )

        return chunks

    # =========================================================
    # OVERSIZED TEXT FALLBACK
    # =========================================================

    def _split_oversized_text(
        self,
        element: DocumentElement,
        hierarchy_path: tuple[str, ...],
    ) -> list[DocumentChunk]:
        """
        Split a single oversized element.

        First attempt:
            sentence-level splitting.

        If one sentence is still too large:
            word-level splitting.

        This guarantees that normal content does not exceed
        max_tokens unless a single token itself is larger than
        the configured limit.
        """

        sentences = self._split_sentences(
            element.content
        )

        chunks: list[DocumentChunk] = []

        current_sentences: list[str] = []

        for sentence in sentences:

            candidate = self._join_content(
                current_sentences
                + [sentence]
            )

            if (
                current_sentences
                and not self._fits_token_limit(
                    candidate
                )
            ):
                chunks.append(
                    self._build_chunk(
                        elements=[element],
                        content=self._join_content(
                            current_sentences
                        ),
                        hierarchy_path=hierarchy_path,
                    )
                )

                current_sentences = []

            # The sentence itself does not fit.
            if not self._fits_token_limit(
                sentence
            ):
                if current_sentences:
                    chunks.append(
                        self._build_chunk(
                            elements=[element],
                            content=self._join_content(
                                current_sentences
                            ),
                            hierarchy_path=hierarchy_path,
                        )
                    )

                    current_sentences = []

                chunks.extend(
                    self._split_oversized_sentence(
                        element=element,
                        sentence=sentence,
                        hierarchy_path=hierarchy_path,
                    )
                )

                continue

            current_sentences.append(
                sentence
            )

        if current_sentences:
            chunks.append(
                self._build_chunk(
                    elements=[element],
                    content=self._join_content(
                        current_sentences
                    ),
                    hierarchy_path=hierarchy_path,
                )
            )

        return chunks

    def _split_oversized_sentence(
        self,
        element: DocumentElement,
        sentence: str,
        hierarchy_path: tuple[str, ...],
    ) -> list[DocumentChunk]:
        """
        Final fallback for a sentence that itself exceeds
        max_tokens.

        Split by words while continuing to respect the token
        budget.
        """

        words = sentence.split()

        chunks: list[DocumentChunk] = []

        current_words: list[str] = []

        for word in words:

            candidate = " ".join(
                current_words + [word]
            )

            if (
                current_words
                and not self._fits_token_limit(
                    candidate
                )
            ):
                chunks.append(
                    self._build_chunk(
                        elements=[element],
                        content=" ".join(
                            current_words
                        ),
                        hierarchy_path=hierarchy_path,
                    )
                )

                current_words = [word]
                continue

            # If one individual word/token exceeds the
            # configured limit, we cannot safely reduce it
            # further with word splitting.
            if not self._fits_token_limit(
                word
            ):
                raise ValueError(
                    "A single token exceeds "
                    f"max_tokens={self.max_tokens}"
                )

            current_words.append(word)

        if current_words:
            chunks.append(
                self._build_chunk(
                    elements=[element],
                    content=" ".join(
                        current_words
                    ),
                    hierarchy_path=hierarchy_path,
                )
            )

        return chunks

    # =========================================================
    # OVERSIZED TABLE ROW
    # =========================================================

    def _split_oversized_table_row(
        self,
        table_element: DocumentElement,
        header: str,
        row: str,
        hierarchy_path: tuple[str, ...],
    ) -> list[DocumentChunk]:
        """
        Handle a row that is itself too large.

        Example:

            Policy | Description | Approval
            Remote Work | very long description | Manager

        becomes:

            Policy: Remote Work
            Description: ...
            Approval: Manager

        This keeps the column -> value relationship explicit
        before token-aware splitting.
        """

        column_names = self._parse_markdown_row(
            header
        )

        values = self._parse_markdown_row(
            row
        )

        column_context: list[str] = []

        for index, value in enumerate(values):

            column_name = (
                column_names[index]
                if index < len(column_names)
                else f"Column {index + 1}"
            )

            column_context.append(
                f"{column_name}: {value}"
            )

        row_content = "\n".join(
            column_context
        )

        # Re-create a DocumentElement using the same
        # identity/version information.
        virtual_element = DocumentElement(
            element_id=table_element.element_id,
            document_id=table_element.document_id,
            document_version_id=(
                table_element.document_version_id
            ),
            page_number=table_element.page_number,
            content_type=ContentType.TABLE,
            content=row_content,
            content_hash=calculate_content_hash(
                row_content
            ),
            metadata={
                **table_element.metadata,
                "table_row_split": True,
            },
        )

        return self._split_oversized_text(
            element=virtual_element,
            hierarchy_path=hierarchy_path,
        )

    # =========================================================
    # DOCUMENT CHUNK CREATION
    # =========================================================

    def _build_chunk(
        self,
        elements: list[DocumentElement],
        content: str,
        hierarchy_path: tuple[str, ...],
    ) -> DocumentChunk:
        """
        Create the final DocumentChunk and preserve lineage.
        """

        if not elements:
            raise ValueError(
                "Cannot create a chunk without elements"
            )

        first_element = elements[0]

        return DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=first_element.document_id,
            document_version_id=(
                first_element.document_version_id
            ),
            element_ids=[
                element.element_id
                for element in elements
            ],
            content=content,
            chunk_index=0,
            content_hash=calculate_content_hash(
                content
            ),
            metadata={
                "content_type": (
                    first_element.content_type.value
                ),
                "hierarchy_path": list(
                    hierarchy_path
                ),
                "page_numbers": sorted(
                    {
                        element.page_number
                        for element in elements
                    }
                ),
            },
        )

    # =========================================================
    # TOKEN / CONTENT HELPERS
    # =========================================================

    def _fits_token_limit(
        self,
        content: str,
    ) -> bool:
        """
        Return True when content is inside the configured
        token budget.
        """

        return (
            self.token_counter(content)
            <= self.max_tokens
        )

    def _join_content(
        self,
        contents: list[str],
    ) -> str:
        """
        Join content pieces while preserving paragraph separation.
        """

        return "\n\n".join(
            content.strip()
            for content in contents
            if content.strip()
        )

    def _split_sentences(
        self,
        content: str,
    ) -> list[str]:
        """
        Basic sentence splitting fallback.

        This can later be replaced with a stronger sentence
        segmentation implementation if evaluation shows
        a quality improvement.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            content.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _parse_markdown_row(
        self,
        row: str,
    ) -> list[str]:
        """
        Convert a Markdown table row into individual cell values.
        """

        return [
            value.strip()
            for value in row.strip()
            .strip("|")
            .split("|")
        ]

    # =========================================================
    # STRUCTURE HELPERS
    # =========================================================

    def _is_structural_element(
        self,
        element: DocumentElement,
    ) -> bool:
        """
        Determine whether the element represents a structural
        document node.
        """

        return element.metadata.get(
            "label"
        ) in {
            "title",
            "section_header",
            "heading",
            "subsection",
        }

    def _get_hierarchy_level(
        self,
        element: DocumentElement,
    ) -> int:
        """
        Read the hierarchy level captured during extraction.
        """

        level = element.metadata.get(
            "hierarchy_level",
            0,
        )

        return (
            level
            if isinstance(level, int)
            else 0
        )