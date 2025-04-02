import nest_asyncio
from llama_cloud_services.parse import ResultType
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_cloud_services import LlamaParse
from llama_index.core.schema import TextNode
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.async_utils import run_jobs, asyncio_run
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import base64
import os
import re
import json
import pickle
import chromadb
from dotenv import load_dotenv


nest_asyncio.apply()
load_dotenv()
os.environ['OPENAI_API_KEY'] = base64.urlsafe_b64decode(os.getenv("OPENAI_API_KEY")).decode('utf-8')
os.environ['LLAMA_CLOUD_API_KEY'] = base64.urlsafe_b64decode(os.getenv("LLAMA_CLOUD_API_KEY")).decode('utf-8')

data_dir = "iclr_docs"


#TODO: retireve automatico dei nomi dei file da dentro la cartella
papers = [
"NewIDP2.docx"
]

embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(model="gpt-4o")

Settings.embed_model = embed_model
Settings.llm = llm

paper_dicts = {}

parser = LlamaParse(result_type=ResultType.MD)

for paper_path in papers:
    paper_base = Path(paper_path).stem
    full_paper_path = str(Path(data_dir) / paper_path)
    md_json_objs = parser.get_json_result("Il tuo percorso assoluto qui")
    # relative_path = os.path.join("iclr_docs", paper_path)
    # md_json_objs = parser.get_json_result(relative_path)
    json_dicts = md_json_objs[0]["pages"]
    paper_dicts[paper_path] = {
        "paper_path": full_paper_path,
        "json_dicts": json_dicts,
    }

# NOTE: these are utility functions to sort the dumped images by the page number
# (they are formatted like "{uuid}-{page_num}.jpg"

def get_page_number(file_name):
    match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0


def _get_sorted_image_files(image_dir):
    """Get image files sorted by page."""
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_page_number)
    return sorted_files

# attach image metadata to the text nodes
def get_text_nodes(json_dicts, paper_path):
    """Split docs into nodes, by separator."""
    nodes = []

    md_texts = [d["md"] for d in json_dicts]

    for idx, md_text in enumerate(md_texts):
        chunk_metadata = {
            "page_num": idx + 1,
            "paper_path": paper_path,
        }
        node = TextNode(
            text=md_text,
            metadata=chunk_metadata,
        )
        nodes.append(node)

    return nodes


# this will combine all nodes from all papers into a single list
all_text_nodes = []
text_nodes_dict = {}
for paper_path, paper_dict in paper_dicts.items():
    json_dicts = paper_dict["json_dicts"]
    text_nodes = get_text_nodes(json_dicts, paper_dict["paper_path"])
    all_text_nodes.extend(text_nodes)
    text_nodes_dict[paper_path] = text_nodes


###############Add Section Metadata##############

class SectionOutput(BaseModel):
    """The metadata for a given section. Includes the section name, title, page that it starts on, and more."""

    section_name: str = Field(
        ..., description="The current section number (e.g. section_name='3.2')"
    )
    section_title: str = Field(
        ...,
        description="The current section title associated with the number (e.g. section_title='Experimental Results')",
    )

    start_page_number: int = Field(..., description="The start page number.")
    is_subsection: bool = Field(
        ...,
        description="True if it's a subsection (e.g. Section 3.2). False if it's not a subsection (e.g. Section 3)",
    )
    description: Optional[str] = Field(
        None,
        description="The extracted line from the source text that indicates this is a relevant section.",
    )

    def get_section_id(self):
        """Get section id."""
        return f"{self.section_name}: {self.section_title}"


class SectionsOutput(BaseModel):
    """A list of all sections."""

    sections: List[SectionOutput]


class ValidSections(BaseModel):
    """A list of indexes, each corresponding to a valid section."""

    valid_indexes: List[int] = Field(
        "List of valid section indexes. Do NOT include sections to remove."
    )

##########Extract into Section Outputs#########

async def aget_sections(
        doc_text: str, llm: Optional[LLM] = None
) -> List[SectionOutput]:
    """Get extracted sections from a provided text."""

    system_prompt = """\
    You are an AI document assistant tasked with extracting out section metadata from a document text. 

- You should ONLY extract out metadata if the document text contains the beginning of a section.
- The metadata schema is listed below - you should extract out the section_name, section_title, start page number, description.
- A valid section MUST begin with a hashtag (#) and have a number (e.g. "1 Introduction" or "Section 1 Introduction"). \
Note: Not all hashtag (#) lines are valid sections. 

- You can extract out multiple section metadata if there are multiple sections on the page. 
- If there are no sections that begin in this document text, do NOT extract out any sections. 
- A valid section MUST be clearly delineated in the document text. Do NOT extract out a section if it is mentioned, \
but is not actually the start of a section in the document text.
- A Figure or Table does NOT count as a section.

    The user will give the document text below.

    """
    llm = llm or OpenAI(model="gpt-4o")
    Settings.embed_model = embed_model
    Settings.llm = llm

    chat_template = ChatPromptTemplate(
        [
            ChatMessage.from_str(system_prompt, "system"),
            ChatMessage.from_str("Document text: {doc_text}", "user"),
        ]
    )
    result = await llm.astructured_predict(
        SectionsOutput, chat_template, doc_text=doc_text
    )
    return result.sections


async def arefine_sections(
        sections: List[SectionOutput], llm: Optional[LLM] = None
) -> List[SectionOutput]:
    """Refine sections based on extracted text."""

    system_prompt = """\
    You are an AI review assistant tasked with reviewing and correcting another agent's work in extracting sections from a document.

    Below is the list of sections with indexes. The sections may be incorrect in the following manner:
    - There may be false positive sections - some sections may be wrongly extracted - you can tell by the sequential order of the rest of the sections
    - Some sections may be incorrectly marked as subsections and vice-versa
    - You can use the description which contains extracted text from the source document to see if it actually qualifies as a section.

    Given this, return the list of indexes that are valid. Do NOT include the indexes to be removed.

    """
    llm = llm or OpenAI(model="gpt-4o")
    Settings.embed_model = embed_model
    Settings.llm = llm

    chat_template = ChatPromptTemplate(
        [
            ChatMessage.from_str(system_prompt, "system"),
            ChatMessage.from_str("Sections in text:\n\n{sections}", "user"),
        ]
    )

    section_texts = "\n".join(
        [f"{idx}: {json.dumps(s.dict())}" for idx, s in enumerate(sections)]
    )

    result = await llm.astructured_predict(
        ValidSections, chat_template, sections=section_texts
    )
    valid_indexes = result.valid_indexes

    new_sections = [s for idx, s in enumerate(sections) if idx in valid_indexes]
    return new_sections


async def acreate_sections(text_nodes_dict):
    sections_dict = {}
    for paper_path, text_nodes in text_nodes_dict.items():
        all_sections = []

        tasks = [aget_sections(n.get_content(metadata_mode="all")) for n in text_nodes]

        async_results = await run_jobs(tasks, workers=8, show_progress=True)
        all_sections = [s for r in async_results for s in r]

        all_sections = await arefine_sections(all_sections)
        sections_dict[paper_path] = all_sections
    return sections_dict

sections_dict = asyncio_run(acreate_sections(text_nodes_dict))
print(sections_dict["NewIDP2.docx"])

# [Optional] SAVE
pickle.dump(sections_dict, open("sections_dict.pkl", "wb"))

# [Optional] LOAD
sections_dict = pickle.load(open("sections_dict.pkl", "rb"))



################### Annotate each chunk with the section metadata ######################


def annotate_chunks_with_sections(chunks, sections):
    main_sections = [s for s in sections if not s.is_subsection]
    # subsections include the main sections too (some sections have no subsections etc.)
    sub_sections = sections

    main_section_idx, sub_section_idx = 0, 0
    for idx, c in enumerate(chunks):
        cur_page = c.metadata["page_num"]
        while (
            main_section_idx + 1 < len(main_sections)
            and main_sections[main_section_idx + 1].start_page_number <= cur_page
        ):
            main_section_idx += 1
        while (
            sub_section_idx + 1 < len(sub_sections)
            and sub_sections[sub_section_idx + 1].start_page_number <= cur_page
        ):
            sub_section_idx += 1

        cur_main_section = main_sections[main_section_idx]
        cur_sub_section = sub_sections[sub_section_idx]

        c.metadata["section_id"] = cur_main_section.get_section_id()
        c.metadata["sub_section_id"] = cur_sub_section.get_section_id()

for paper_path, text_nodes in text_nodes_dict.items():
    sections = sections_dict[paper_path]
    annotate_chunks_with_sections(text_nodes, sections)


# SAVE
pickle.dump(text_nodes_dict,open("iclr_text_nodes.pkl", "wb"))

# LOAD
text_nodes_dict = pickle.load(open("iclr_text_nodes.pkl", "rb"))


all_text_nodes = []
for paper_path, text_nodes in text_nodes_dict.items():
    all_text_nodes.extend(text_nodes)

print(len(all_text_nodes))


#########Build Indexes######

persist_dir = "storage_chroma"

vector_store = ChromaVectorStore.from_params(
    collection_name="text_nodes", persist_dir=persist_dir
)
index = VectorStoreIndex.from_vector_store(vector_store,embed_model=embed_model)

###################################
# Don't run the block below if you've already inserted the nodes. Only run if it's your first time!!
################################
index.insert_nodes(all_text_nodes)

##############################################














