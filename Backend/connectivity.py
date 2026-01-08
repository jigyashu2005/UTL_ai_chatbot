import requests
import logging

def check_connectivity(url='https://www.google.com', timeout=5):
    """
    Checks internet connectivity by making a request to a reliable URL.
    
    Args:
        url (str): The URL to check against.
        timeout (int): The timeout in seconds.
        
    Returns:
        bool: True if connected, False otherwise.
    """
    try:
        logging.info(f"Checking connectivity to {url}...")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        logging.info("Connectivity check passed.")
        return True
    except requests.RequestException as e:
        logging.error(f"Connectivity check failed: {e}")
        return False
