from googleapiclient.discovery import build
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

youtube = build('youtube', 'v3', developerKey="AIzaSyA5AsORnkuR2Wj0xsS2vKFtwZ5iHCgVx1Y")
chrome_path = 'C:\selenium\chromedriver.exe'
browser_option = webdriver.ChromeOptions()
browser_option.add_argument('--no-sandbox')
browser_option.add_argument('--headless')
browser_option.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=browser_option)


