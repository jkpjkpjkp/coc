"""search google.

currently supports only text.
TODO: add image search (text -> image)
"""

import os
from googleapiclient.discovery import build

def _google_search(search_term, api_key, cse_id, **kwargs):
    """Perform a Google search using the Custom Search API.

    Args:
        search_term: The search query
        api_key: Your API key
        cse_id: Your Custom Search Engine ID
        **kwargs: Additional parameters for the search

    Returns:
        A dictionary containing the search results
    """

    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'

    service = build("customsearch", "v1", developerKey=api_key)
    results = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return results

# Example usage
api_key = "AIzaSyAlbt1KiIQt9II7ukYslbC08zfrsK5Qx_c"
cse_id = "a663af489502947ee"
query = "python programming"

def google_search(search_term, **kwargs):
    return _google_search(search_term, api_key, cse_id, **kwargs)

if __name__ == "__main__":
    # Perform the search
    results = _google_search(query, api_key, cse_id)

    # Process and display the results
    for item in results.get('items', []):
        print(f"Title: {item['title']}")
        print(f"Link: {item['link']}")
        print(f"Snippet: {item['snippet']}")
        print("---")