"""
Academic paper search - PubMed, arXiv, CrossRef
No web search needed - these APIs cover everything!
"""

import time
import requests
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
import arxiv
from Bio import Entrez

from models import Paper


class PubMedSearcher:
    """Search PubMed (biomedical papers)"""
    
    def __init__(self, email: str = "shweta.shankar@students.iiserpune.ac.in"):
        Entrez.email = email
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search PubMed"""
        papers = []
        
        try:
            # Search
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()
            
            ids = record.get("IdList", [])
            if not ids:
                return papers
            
            # Fetch details
            handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml")
            records = Entrez.read(handle)
            handle.close()
            
            # Parse
            for article in records.get('PubmedArticle', []):
                try:
                    medline = article['MedlineCitation']
                    article_data = medline['Article']
                    
                    # Authors
                    authors = []
                    if 'AuthorList' in article_data:
                        for author in article_data['AuthorList'][:10]:
                            if 'LastName' in author and 'Initials' in author:
                                authors.append(f"{author['LastName']} {author['Initials']}")
                    
                    # Year
                    year = "Unknown"
                    if 'Journal' in article_data:
                        pub_date = article_data['Journal']['JournalIssue'].get('PubDate', {})
                        year = str(pub_date.get('Year', 'Unknown'))
                    
                    # Abstract
                    abstract = ""
                    if 'Abstract' in article_data:
                        abstract_parts = article_data['Abstract'].get('AbstractText', [])
                        abstract = " ".join([str(text) for text in abstract_parts])
                    
                    # DOI
                    doi = None
                    if 'ELocationID' in article_data:
                        for eloc in article_data['ELocationID']:
                            if hasattr(eloc, 'attributes') and eloc.attributes.get('EIdType') == 'doi':
                                doi = str(eloc)
                    
                    paper = Paper(
                        title=str(article_data.get('ArticleTitle', 'No Title')),
                        authors=authors if authors else ["Unknown"],
                        year=year,
                        abstract=abstract,
                        source="PubMed",
                        doi=doi,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{medline['PMID']}/",
                        journal=article_data.get('Journal', {}).get('Title', 'Unknown')
                    )
                    
                    papers.append(paper)
                
                except Exception as e:
                    print(f"⚠️  Error parsing article: {e}")
                    continue
        
        except Exception as e:
            print(f"⚠️  PubMed search error: {e}")
        
        return papers


class ArxivSearcher:
    """Search arXiv (preprints)"""
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search arXiv"""
        papers = []
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    year=str(result.published.year),
                    abstract=result.summary,
                    source="arXiv",
                    url=result.entry_id,
                    journal=f"arXiv:{result.primary_category}"
                )
                papers.append(paper)
        
        except Exception as e:
            print(f"⚠️  arXiv search error: {e}")
        
        return papers


class CrossRefSearcher:
    """Search CrossRef (general academic)"""
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search CrossRef"""
        papers = []
        
        try:
            url = "https://api.crossref.org/works"
            params = {
                "query": query,
                "rows": max_results,
                "sort": "relevance"
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('message', {}).get('items', []):
                    # Authors
                    authors = []
                    if 'author' in item:
                        for author in item['author'][:10]:
                            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                            if name:
                                authors.append(name)
                    
                    # Year
                    year = "Unknown"
                    if 'published-print' in item:
                        year = str(item['published-print']['date-parts'][0][0])
                    elif 'published-online' in item:
                        year = str(item['published-online']['date-parts'][0][0])
                    
                    # Abstract
                    abstract = item.get('abstract', 'No abstract available')
                    
                    paper = Paper(
                        title=item.get('title', ['No Title'])[0] if isinstance(item.get('title'), list) else 'No Title',
                        authors=authors if authors else ["Unknown"],
                        year=year,
                        abstract=abstract,
                        source="CrossRef",
                        doi=item.get('DOI'),
                        url=item.get('URL'),
                        journal=item.get('container-title', ['Unknown'])[0] if item.get('container-title') else 'Unknown'
                    )
                    
                    papers.append(paper)
        
        except Exception as e:
            print(f"⚠️  CrossRef search error: {e}")
        
        return papers


class PaperSearcher:
    """Search all academic sources"""
    
    def __init__(self, email: str = "user@example.com"):
        self.pubmed = PubMedSearcher(email)
        self.arxiv = ArxivSearcher()
        self.crossref = CrossRefSearcher()
        
    def search(self, query: str, max_per_source: int = 50) -> List[Paper]:
        """Search all sources - now with higher limits!"""
        print(f"\n🔍 Searching for: '{query}' (max {max_per_source} per source)")
        print("─" * 60)
        
        all_papers = []
    
        # PubMed (Entrez allows up to 100-500; we'll cap at input)
        print("📚 PubMed...", end=" ", flush=True)
        pubmed_papers = self.pubmed.search(query, min(max_per_source, 100))  # Soft cap for stability
        all_papers.extend(pubmed_papers)
        print(f"✓ {len(pubmed_papers)} papers")
        time.sleep(0.5)  # Gentle rate limit
    
        # arXiv (API handles 100+ fine)
        print("📄 arXiv...", end=" ", flush=True)
        arxiv_papers = self.arxiv.search(query, max_per_source)
        all_papers.extend(arxiv_papers)
        print(f"✓ {len(arxiv_papers)} papers")
        time.sleep(0.5)
    
        # CrossRef (API rows up to 1000, but we cap at input)
        print("🔬 CrossRef...", end=" ", flush=True)
        crossref_papers = self.crossref.search(query, min(max_per_source, 200))
        all_papers.extend(crossref_papers)
        print(f"✓ {len(crossref_papers)} papers")
    
        print(f"\n✅ Total before deduplication: {len(all_papers)} papers")
        
        # Deduplicate (as before)
        unique = {}
        for paper in all_papers:
            key = paper.doi if paper.doi else paper.title.lower().strip()
            if key not in unique:
                unique[key] = paper
        deduplicated = list(unique.values())
        print(f"✅ After deduplication: {len(deduplicated)} papers")
    
        # Relevance sort + filter (SAFE VERSION)
        def score_paper(paper: Paper, query: str) -> float:
            query_words = set(query.lower().split())
            abstract_words = set(paper.abstract.lower().split()) if paper.abstract else set()
            relevance = len(query_words.intersection(abstract_words)) / max(len(query_words), 1)
            try:
                year = int(paper.year)
                recency = 1.0 if year >= 2020 else 0.5
            except (ValueError, TypeError):  # Handles 'Unknown', None, etc.
                recency = 0.5  # Neutral for unknowns
            return relevance * recency
    
        deduplicated.sort(key=lambda p: score_paper(p, query), reverse=True)
    
        # Clean: Skip empty abstracts, but keep if year/abstract missing (now safe)
        cleaned = [p for p in deduplicated if (p.abstract and len(p.abstract.strip()) > 20) or p.title]  # Looser filter
        #if len(cleaned) > 50:
         #   print(f"⚠️  {len(cleaned)} papers fetched—consider slicing to top 30-50 in prompts for model efficiency.")
            
        print("─" * 60)
        return cleaned  # All good now!
