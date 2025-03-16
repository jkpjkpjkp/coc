import os
import unittest
from googleapiclient.errors import HttpError
from coc.tool.google_search.mod import _google_search  # Replace 'your_module' with the actual module name

class TestGoogleSearch(unittest.TestCase):
    def setUp(self):
        """Save the original environment variables before each test."""
        self.original_http_proxy = os.environ.get('HTTP_PROXY')
        self.original_https_proxy = os.environ.get('HTTPS_PROXY')
        self.original_all_proxy = os.environ.get('ALL_PROXY')

    def tearDown(self):
        """Restore the original environment variables after each test."""
        if self.original_http_proxy is not None:
            os.environ['HTTP_PROXY'] = self.original_http_proxy
        else:
            os.environ.pop('HTTP_PROXY', None)
        if self.original_https_proxy is not None:
            os.environ['HTTPS_PROXY'] = self.original_https_proxy
        else:
            os.environ.pop('HTTPS_PROXY', None)
        if self.original_all_proxy is not None:
            os.environ['ALL_PROXY'] = self.original_all_proxy
        else:
            os.environ.pop('ALL_PROXY', None)

    def test_valid_search(self):
        """Test _google_search with valid inputs and verify the result structure."""
        api_key = "AIzaSyAlbt1KiIQt9II7ukYslbC08zfrsK5Qx_c"  # Replace with a valid API key
        cse_id = "a663af489502947ee"  # Replace with a valid CSE ID
        search_term = "python programming"

        results = _google_search(search_term, api_key, cse_id)

        # Check the structure of the returned dictionary
        self.assertIsInstance(results, dict, "Results should be a dictionary")
        self.assertIn('items', results, "Results should contain 'items' key")
        self.assertIsInstance(results['items'], list, "'items' should be a list")

        # If there are items, check their structure
        if results['items']:
            for item in results['items']:
                self.assertIn('title', item, "Each item should have a 'title'")
                self.assertIn('link', item, "Each item should have a 'link'")
                self.assertIn('snippet', item, "Each item should have a 'snippet'")

        # Verify environment variables are set
        self.assertEqual(os.environ['HTTP_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['HTTPS_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['ALL_PROXY'], 'http://127.0.0.1:7890')

    def test_additional_params(self):
        """Test _google_search with additional parameters."""
        api_key = "AIzaSyAlbt1KiIQt9II7ukYslbC08zfrsK5Qx_c"  # Replace with a valid API key
        cse_id = "a663af489502947ee"  # Replace with a valid CSE ID
        search_term = "python programming"

        results = _google_search(search_term, api_key, cse_id, num=3)

        # Check the structure of the returned dictionary
        self.assertIsInstance(results, dict, "Results should be a dictionary")
        self.assertIn('items', results, "Results should contain 'items' key")
        self.assertIsInstance(results['items'], list, "'items' should be a list")

        # Verify environment variables are set
        self.assertEqual(os.environ['HTTP_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['HTTPS_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['ALL_PROXY'], 'http://127.0.0.1:7890')

    def test_invalid_api_key(self):
        """Test _google_search with an invalid API key."""
        api_key = "invalid_key"  # Intentionally invalid
        cse_id = "a663af489502947ee"  # Replace with a valid CSE ID
        search_term = "python programming"

        # Expect an HttpError due to invalid API key
        with self.assertRaises(HttpError):
            _google_search(search_term, api_key, cse_id)

        # Verify environment variables are still set even if the API call fails
        self.assertEqual(os.environ['HTTP_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['HTTPS_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['ALL_PROXY'], 'http://127.0.0.1:7890')

    def test_invalid_cse_id(self):
        """Test _google_search with an invalid CSE ID."""
        api_key = "AIzaSyAlbt1KiIQt9II7ukYslbC08zfrsK5Qx_c"  # Replace with a valid API key
        cse_id = "invalid_cse"  # Intentionally invalid
        search_term = "python programming"

        # Expect an HttpError due to invalid CSE ID
        with self.assertRaises(HttpError):
            _google_search(search_term, api_key, cse_id)

        # Verify environment variables are still set even if the API call fails
        self.assertEqual(os.environ['HTTP_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['HTTPS_PROXY'], 'http://127.0.0.1:7890')
        self.assertEqual(os.environ['ALL_PROXY'], 'http://127.0.0.1:7890')

if __name__ == '__main__':
    unittest.main()