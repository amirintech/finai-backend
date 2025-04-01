"""SEC Filing Retriever module."""

from typing import Dict, Optional, List
from sec_api import QueryApi, ExtractorApi, RenderApi
from datetime import datetime


class SECFilingRetriever:
    """A class to retrieve and process SEC filings using the sec-api library."""

    def __init__(self, api_key):
        self.query_api = QueryApi(api_key=api_key)
        self.extractor_api = ExtractorApi(api_key=api_key)
        self.render_api = RenderApi(api_key=api_key)
        self.filing_cache = {}

    def get_filing_by_year(self, ticker: str, year: int, form_type: str = "10-K") -> Dict:
        """
        Get the filing of a specific type for a given ticker and year.

        Args:
            ticker: The stock ticker symbol (e.g., AAPL, MSFT)
            year: The fiscal year to retrieve (e.g., 2023)
            form_type: The type of form to retrieve (e.g., "10-K", "10-Q", "8-K")

        Returns:
            A dictionary containing the filing metadata
        """
        ticker = ticker.upper()

        # check cache first
        cache_key = f"{ticker}_{form_type}_{year}"
        if cache_key in self.filing_cache:
            return self.filing_cache[cache_key]

        try:
            # Convert year to date range
            start_date = f"{year}-01-01"
            end_date = f"{year + 1}-12-31"  # Look a bit beyond to catch fiscal year filings

            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker} AND formType:\"{form_type}\" AND filedAt:[{start_date} TO {end_date}]"
                    }
                },
                "from": 0,
                "size": 5,  # Get more to find the right period
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            response = self.query_api.get_filings(query)

            if not response['filings']:
                raise ValueError(f"No {form_type} filings found for {ticker} in {year}")

            # Find the filing that best matches the requested year
            matching_filing = None
            for filing in response['filings']:
                # Check the period of report
                if 'periodOfReport' in filing:
                    report_date = filing['periodOfReport']
                    report_year = datetime.strptime(report_date, "%Y-%m-%d").year
                    if report_year == year:
                        matching_filing = filing
                        break

            # If no exact match, use the first (most recent) filing
            if matching_filing is None:
                matching_filing = response['filings'][0]
                print(f"Warning: No exact {year} match found for {ticker}. Using filing from {matching_filing.get('periodOfReport', 'unknown date')}")

            self.filing_cache[cache_key] = matching_filing
            return matching_filing

        except Exception as e:
            raise Exception(f"Error fetching {form_type} filing for {ticker} in {year}: {str(e)}")

    def get_latest_filing(self, ticker: str, form_type: str = "10-K") -> Dict:
        """
        Get the latest filing of a specific type for a given ticker.

        Args:
            ticker: The stock ticker symbol (e.g., AAPL, MSFT)
            form_type: The type of form to retrieve (e.g., "10-K", "10-Q", "8-K")

        Returns:
            A dictionary containing the filing metadata
        """
        ticker = ticker.upper()

        # check cache first
        cache_key = f"{ticker}_{form_type}_latest"
        if cache_key in self.filing_cache:
            return self.filing_cache[cache_key]

        try:
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker} AND formType:\"{form_type}\""
                    }
                },
                "from": 0,
                "size": 1,
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            response = self.query_api.get_filings(query)

            if not response['filings']:
                raise ValueError(f"No {form_type} filings found for {ticker}")

            filing = response['filings'][0]
            self.filing_cache[cache_key] = filing
            
            # Extract year from periodOfReport for reference
            if 'periodOfReport' in filing:
                report_year = datetime.strptime(filing['periodOfReport'], "%Y-%m-%d").year
                # Also cache by specific year
                year_cache_key = f"{ticker}_{form_type}_{report_year}"
                self.filing_cache[year_cache_key] = filing
                
            return filing

        except Exception as e:
            raise Exception(f"Error fetching {form_type} filing for {ticker}: {str(e)}")

    def get_available_years(self, ticker: str, form_type: str = "10-K", max_years: int = 5) -> List[int]:
        """
        Get a list of years for which filings are available.

        Args:
            ticker: The stock ticker symbol
            form_type: The form type to check
            max_years: Maximum number of years to return

        Returns:
            List of years with available filings (most recent first)
        """
        ticker = ticker.upper()
        
        try:
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{ticker} AND formType:\"{form_type}\""
                    }
                },
                "from": 0,
                "size": max_years,
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            response = self.query_api.get_filings(query)
            
            if not response['filings']:
                return []
                
            years = []
            for filing in response['filings']:
                if 'periodOfReport' in filing:
                    report_year = datetime.strptime(filing['periodOfReport'], "%Y-%m-%d").year
                    years.append(report_year)
                    
                    # Cache the filing by year for future use
                    cache_key = f"{ticker}_{form_type}_{report_year}"
                    self.filing_cache[cache_key] = filing
                    
            return sorted(list(set(years)), reverse=True)  # Remove duplicates and sort
            
        except Exception as e:
            print(f"Error fetching available years for {ticker}: {str(e)}")
            return []

    def get_filing_content(self, filing: Dict, content_type: str = "text") -> str:
        """
        Extract content from an SEC filing.

        Args:
            filing: Filing metadata dictionary from get_latest_filing
            content_type: Type of content to retrieve ('text', 'html', or 'raw')

        Returns:
            The content of the filing in the specified format
        """
        try:
            accession_number = filing['accessionNo']

            if content_type == "text":
                return self.extractor_api.get_filing_text(accession_number)

            elif content_type == "html":
                return self.render_api.get_filing(accession_number)

            elif content_type == "raw":
                return self.extractor_api.get_filing(accession_number)

            else:
                raise ValueError(f"Invalid content_type: {content_type}. Must be 'text', 'html', or 'raw'.")

        except Exception as e:
            raise Exception(f"Error extracting filing content: {str(e)}")

    def get_section_content(self, filing: Dict, section: str) -> str:
        """
        Extract a specific section from a filing (like Item 1A Risk Factors).

        Args:
            filing: Filing metadata dictionary from get_latest_filing
            section: Section name to extract (e.g., "1A", "7", "7A", "1", "2")

        Returns:
            The content of the specified section
        """
        try:
            # Section options include: "1", "1A", "1B", "2", "3", "4", "5", "6",
            # "7", "7A", "8", "9", "9A", "9B", "10", "11", "12", "13", "14", "15"
            url = self._get_sec_filing_url(filing['cik'], filing['accessionNo'])
            return self.extractor_api.get_section(url, section)

        except Exception as e:
            raise Exception(f"Error extracting section {section}: {str(e)}")

    def _get_sec_filing_url(self, cik: str, accession_number: str):
        accession_number_no_dashes = accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_no_dashes}/index.json"

    def get_filing_year(self, filing: Dict) -> Optional[int]:
        """
        Extract the year from a filing's period of report.
        
        Args:
            filing: Filing metadata dictionary
            
        Returns:
            Year as integer or None if not available
        """
        if 'periodOfReport' in filing:
            try:
                return datetime.strptime(filing['periodOfReport'], "%Y-%m-%d").year
            except:
                pass
        return None