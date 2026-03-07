import arxiv
import os
from tqdm import tqdm

DOWNLOAD_DIR = "data/raw_papers"

def search_papers(query, max_results=5):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in search.results():
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d")
        })

    return papers


def download_papers(papers):

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for paper in tqdm(papers):

        pdf_url = paper["pdf_url"]
        title = paper["title"].replace(" ", "_").replace("/", "")

        filename = f"{title}.pdf"
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        if os.path.exists(filepath):
            continue

        client = arxiv.Client()
        search = arxiv.Search(id_list=[pdf_url.split("/")[-1]])

        for result in client.results(search):
            result.download_pdf(dirpath=DOWNLOAD_DIR, filename=filename)

        paper["file_path"] = filepath

    return papers