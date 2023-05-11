import json
import urllib
import urllib.error
import urllib.parse
import urllib.request

REQUEST_DEPTH_LIMIT = 10000

# filters
ROOT = "itemListElement"
DESC_KEY = "detailedDescription"
RESULT_KEY = "result"
URL_KEY = "url"


class GoogleSearchError(Exception):
    pass


def google_search(query, semaphore, api_key=None, limit=1, indent=True):
    """
    Search Google Knowledge Graph utility method
    Args:
        api_key: private Google Cloud API key
        query: keyword to search
        limit: requests depth limit
        indent: response json formatting
        semaphore: limit resource access on concurrent API calls
    Returns:
        Response from graph in raw json form
    """
    with semaphore:
        urls = []
        if api_key is None:
            raise GoogleSearchError(
                "Cannot validate identify because API_KEY variable is None."
            )
        params = google_api_params(
            query=query, api_key=api_key, limit=limit, indent=indent
        )
        url = build_url(params)
        try:
            response = do_GET(url)
        except GoogleSearchError:
            return urls
        try:
            for element in response[ROOT]:
                urls.append(element[RESULT_KEY][DESC_KEY][URL_KEY])
            return urls
        except KeyError:
            return urls


def google_api_params(**kwargs):
    return {
        "query": kwargs.get("query"),
        "limit": int(kwargs.get("limit", 10)),
        "indent": kwargs.get("indent", True),
        "key": kwargs.get("api_key"),
    }


def build_url(params):
    service_url = "https://kgsearch.googleapis.com/v1/entities:search"
    url = service_url + "?" + urllib.parse.urlencode(params)
    return url


def do_GET(url):
    try:
        response = json.loads(urllib.request.urlopen(url).read())
        return response
    except urllib.error.HTTPError as e:
        raise GoogleSearchError(e)
