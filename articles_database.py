#!/usr/bin/env python3

from Bio import Entrez
from Bio import Medline
import pandas as pd

def search(year: int, email: str, query: str, num_articles: int) -> list:
    """Search for articles in PubMed based on keywords.

    Args:
        year: Year used to specify the mindate and maxdate parameter needed to define the date range that limits the search result.
        email: Your email so that the NCBI can contact you if there is a problem.
        query: The text query you wish to search for.
        num_articles: Total number of unique identifiers (UIDs) from the retrieved set to be shown in the output (maximum of 10,000 UIDs).

    Returns:
        idlist: List of PubMed article UIDs.
    """
    mindate = str(year) + '/1/1'
    maxdate = str(year) + '/12/31'
    Entrez.email = email
    handle = Entrez.esearch(db = "pubmed", term = query, retmax = num_articles, sort = "pub_date", usehistory = "y", mindate = mindate, maxdate = maxdate)
    results = Entrez.read(handle)
    print('There are {} articles associated with the term "{}" that were published during {}. You are going to fetch {} maximum.'.format(results['Count'], query, year, num_articles))
    idlist = results["IdList"]
    handle.close()
    return idlist

def fetch(idlist: list) -> list:
    """Fetch Pubmed articles using a list of UIDs.
    
    Args:
        idlist: List of PubMed article UIDs.
    Returns:
        records: List of dictionaries with information about articles. Each dictionary contains PubMed MEDLINE display elements. For a description of each element, see https://www.nlm.nih.gov/bsd/mms/medlineelements.html.
    """
    handle = Entrez.efetch(db = "pubmed", id = idlist, rettype = "medline", retmode = "text")
    records = list(Medline.parse(handle))
    print('Articles fetched.')
    return records

def database(records: list, fields_list: list) -> pd.DataFrame:
    """Build a database with relevant information about PubMed articles.

    Args:
        records: List of dictionaries with information about articles. Each dictionary contains PubMed MEDLINE display elements. For a description of each element, see https://www.nlm.nih.gov/bsd/mms/medlineelements.html.
        fields_list: List of the PubMed MEDLINE display elements of interest.
    Returns:
        article_db: Database with PubMed MEDLINE display elements of interest.
    """
    df = pd.DataFrame(records)
    articles_db = df[fields_list]
    print('Your database is ready! It has {} rows and {} columns.'.format(articles_db.shape[0], articles_db.shape[1]))
    return articles_db

if __name__ == "__main__":
    articles = []
    for year in range(2018, 2022): #years used to limit the search
        idlist = search(year, "example@email.com", "artificial intelligence", 10000)
        records = fetch(idlist)
        articles.extend(records)
    fields_list = ['AB', 'AD', 'AU', 'DP', 'TA', 'JT', 'PL', 'PT', 'PMID', 'TI'] #fields used to create the database
    df = database(articles, fields_list)
    df.to_csv('data/data.csv', index = False)
