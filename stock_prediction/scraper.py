"""
Selenium-based Yahoo Finance real-time stock data scraper.
Uses headless Chrome to scrape current stock prices and key metrics.
Includes yfinance fallback for reliability.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_driver():
    """Create headless Chrome WebDriver instance with anti-detection."""
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--lang=id-ID')
    # More realistic user-agent
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')
    options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=options)
        # Remove webdriver flag
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['id-ID', 'id', 'en-US', 'en']});
                window.chrome = {runtime: {}};
            '''
        })
        driver.set_page_load_timeout(30)
        return driver
    except WebDriverException as e:
        logger.error(f"[Selenium] Failed to create driver: {e}")
        return None


def _safe_find(driver, selector, timeout=5):
    """Safely find element with timeout."""
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    try:
        elem = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        return elem.text.strip()
    except (TimeoutException, NoSuchElementException):
        return None


def _safe_find_value(driver, selector, timeout=5):
    """Safely find element and get its value attribute."""
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    
    try:
        elem = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        val = elem.get_attribute('value')
        if val:
            return val.strip()
        return elem.text.strip()
    except (TimeoutException, NoSuchElementException):
        return None


def _parse_number(text):
    """Parse number string to float, handling commas and special chars."""
    if not text:
        return None
    try:
        cleaned = text.replace(',', '').replace('%', '').replace('+', '').strip()
        if 'B' in cleaned:
            return float(cleaned.replace('B', '')) * 1_000_000_000
        elif 'M' in cleaned:
            return float(cleaned.replace('M', '')) * 1_000_000
        elif 'K' in cleaned:
            return float(cleaned.replace('K', '')) * 1_000
        elif 'T' in cleaned:
            return float(cleaned.replace('T', '')) * 1_000_000_000_000
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def _scrape_with_js(driver):
    """Fallback: use JavaScript to extract data from page."""
    try:
        js_code = '''
        (function() {
            var result = {};
            // Try to get price from various sources
            var priceEl = document.querySelector('[data-symbol] [data-field="regularMarketPrice"]') ||
                         document.querySelector('fin-streamer[data-field="regularMarketPrice"]');
            if (priceEl) result.price = priceEl.textContent.trim();
            
            // Get all fin-streamer elements
            var streamers = document.querySelectorAll('fin-streamer[data-field]');
            streamers.forEach(function(s) {
                var field = s.getAttribute('data-field');
                var val = s.textContent.trim();
                result[field] = val;
            });
            
            // Try data-test selectors
            var qspPrice = document.querySelector('[data-test="qsp-price"]');
            if (qspPrice) result.qsp_price = qspPrice.textContent.trim();
            
            return JSON.stringify(result);
        })()
        '''
        result_json = driver.execute_script(js_code)
        if result_json:
            import json
            return json.loads(result_json)
    except Exception as e:
        logger.debug(f"[Selenium] JS fallback failed: {e}")
    return {}


def scrape_yahoo_finance(stock_code, driver=None):
    """
    Scrape real-time stock data from Yahoo Finance using Selenium.
    
    Args:
        stock_code: Stock ticker (e.g., 'BBCA.JK')
        driver: Optional existing WebDriver instance
    
    Returns:
        dict with keys: price, change, change_pct, open, high, low, volume,
                       prev_close, market_cap, pe_ratio, timestamp
        None if scraping fails
    """
    close_driver = driver is None
    
    if driver is None:
        driver = create_driver()
    
    if driver is None:
        logger.error("[Selenium] Cannot create driver, trying yfinance fallback")
        return _yfinance_fallback(stock_code)
    
    try:
        url = f"https://finance.yahoo.com/quote/{stock_code}"
        logger.info(f"[Selenium] Navigating to {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(4)
        
        # Try to dismiss cookie consent dialog (multiple selectors)
        for consent_sel in ['button[name="agree"]', '.consent-banner .accept-btn', '#accept-all-btn', '[data-testid="accept-btn"]']:
            try:
                consent = driver.find_element(By.CSS_SELECTOR, consent_sel)
                consent.click()
                time.sleep(1)
                break
            except NoSuchElementException:
                continue
        
        # Additional wait for dynamic content
        time.sleep(2)
        
        data = {}
        
        # Current Price - try multiple selectors
        price_text = (
            _safe_find(driver, f'fin-streamer[data-symbol="{stock_code}"][data-field="regularMarketPrice"]', 5) or
            _safe_find(driver, 'fin-streamer[data-field="regularMarketPrice"]', 3) or
            _safe_find(driver, '[data-test="qsp-price"]', 3) or
            _safe_find(driver, '.price-current', 2)
        )
        data['price'] = _parse_number(price_text)
        
        # If still no price, try JS fallback
        if data['price'] is None:
            logger.info(f"[Selenium] Standard selectors failed, trying JS fallback for {stock_code}")
            js_data = _scrape_with_js(driver)
            if js_data:
                data['price'] = _parse_number(js_data.get('price') or js_data.get('qsp_price'))
        
        # If still no price, try yfinance fallback
        if data['price'] is None:
            logger.warning(f"[Selenium] All selectors failed for {stock_code}, using yfinance fallback")
            return _yfinance_fallback(stock_code)
        
        # Price Change
        change_text = (
            _safe_find(driver, f'fin-streamer[data-symbol="{stock_code}"][data-field="regularMarketChange"]', 3) or
            _safe_find(driver, 'fin-streamer[data-field="regularMarketChange"]', 2) or
            _safe_find(driver, '[data-test="qsp-price-change"]', 2)
        )
        data['change'] = _parse_number(change_text)
        
        # Price Change Percent
        pct_text = (
            _safe_find(driver, f'fin-streamer[data-symbol="{stock_code}"][data-field="regularMarketChangePercent"]', 3) or
            _safe_find(driver, 'fin-streamer[data-field="regularMarketChangePercent"]', 2) or
            _safe_find(driver, '[data-test="qsp-price-change-percent"]', 2)
        )
        data['change_pct'] = _parse_number(pct_text)
        
        # Other metrics using data-field attributes
        fields = {
            'open': 'regularMarketOpen',
            'high': 'regularMarketDayHigh',
            'low': 'regularMarketDayLow',
            'volume': 'regularMarketVolume',
            'prev_close': 'regularMarketPreviousClose',
            'market_cap': 'marketCap',
            'pe_ratio': 'trailingPE',
            'fifty_two_week_high': 'fiftyTwoWeekHigh',
            'fifty_two_week_low': 'fiftyTwoWeekLow',
            'avg_volume': 'averageDailyVolume3Month',
        }
        
        for key, field in fields.items():
            text = _safe_find(driver, f'fin-streamer[data-field="{field}"]', 2)
            if text is None and js_data:
                text = js_data.get(field)
            data[key] = _parse_number(text)
        
        # Timestamp
        from datetime import datetime
        data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data['stock_code'] = stock_code
        
        logger.info(f"[Selenium] Successfully scraped {stock_code}: price={data['price']}")
        return data
        
    except Exception as e:
        logger.error(f"[Selenium] Error scraping {stock_code}: {e}")
        return _yfinance_fallback(stock_code)
    
    finally:
        if close_driver:
            try:
                driver.quit()
            except Exception:
                pass


def _yfinance_fallback(stock_code):
    """Fallback to yfinance when Selenium fails."""
    try:
        import yfinance as yf
        from datetime import datetime
        
        logger.info(f"[yfinance] Using fallback for {stock_code}")
        ticker = yf.Ticker(stock_code)
        info = ticker.info
        
        if not info or 'regularMarketPrice' not in info:
            logger.warning(f"[yfinance] No data for {stock_code}")
            return None
        
        # Get today's data if market is open
        hist = ticker.history(period="1d")
        
        data = {
            'stock_code': stock_code,
            'price': info.get('regularMarketPrice'),
            'change': info.get('regularMarketChange'),
            'change_pct': info.get('regularMarketChangePercent'),
            'open': info.get('regularMarketOpen') or (hist['Open'].iloc[0] if len(hist) > 0 else None),
            'high': info.get('regularMarketDayHigh') or (hist['High'].iloc[0] if len(hist) > 0 else None),
            'low': info.get('regularMarketDayLow') or (hist['Low'].iloc[0] if len(hist) > 0 else None),
            'volume': info.get('regularMarketVolume') or (hist['Volume'].iloc[0] if len(hist) > 0 else None),
            'prev_close': info.get('regularMarketPreviousClose') or (hist['Close'].iloc[0] if len(hist) > 0 else None),
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'avg_volume': info.get('averageDailyVolume3Month'),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        logger.info(f"[yfinance] Successfully fetched {stock_code}: price={data['price']}")
        return data
        
    except Exception as e:
        logger.error(f"[yfinance] Fallback failed for {stock_code}: {e}")
        return None


def scrape_multiple(stock_codes):
    """
    Scrape multiple stocks using a single browser session.
    
    Args:
        stock_codes: List of stock ticker symbols
    
    Returns:
        dict mapping stock_code to data dict
    """
    results = {}
    driver = None
    
    try:
        driver = create_driver()
        if driver is None:
            # Fallback to yfinance for all
            for code in stock_codes:
                data = _yfinance_fallback(code)
                if data:
                    results[code] = data
            return results
            
        for code in stock_codes:
            data = scrape_yahoo_finance(code, driver=driver)
            if data:
                results[code] = data
            time.sleep(2)  # Rate limiting between requests
    except Exception as e:
        logger.error(f"[Selenium] Error in scrape_multiple: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
    
    return results
