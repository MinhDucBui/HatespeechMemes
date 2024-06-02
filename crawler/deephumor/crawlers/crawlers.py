import os
import time
from lxml import html, etree
from selenium.webdriver.chrome.options import Options
from .utils import time_to_str, load_image
from deephumor.data import SPECIAL_TOKENS
from deephumor.data.utils import clean_text, check_text
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By

HTML_PATH = "/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/htmls"



def crawl_templates(page=1):
    """Crawls templates from All-time page.

    Args:
        page (int): page number
    """

    meme_templates = []
    links = [#"https://memegenerator.net/Ordinary-Muslim-Man",]
             #"https://memegenerator.net/Asian-College-Freshman"]
             #"https://memegenerator.net/Sassy-Black-Woman",
             #"https://memegenerator.net/Scumbag-Catholic-Priest"]
            # "https://memegenerator.net/Scumbag-Whitehouse",
            # "https://memegenerator.net/Scumbag-God",
            # "https://memegenerator.net/Scumbag-America",
            # "https://memegenerator.net/Angry-Muslim-Guy",
            # "https://memegenerator.net/Confused-Muslim-Girl",
            # "https://memegenerator.net/Stereotypical-Redneck",
             "https://memegenerator.net/Stereotypical-Indian-Telemarketer"]
    """
    try:sasdasd
        r = requests.get(url)
        tree = html.fromstring(r.content)
        divs = tree.xpath('//div[@class="char-img"]/a')

        for div in divs:
            link = div.get('href')
            img = div.find('img')
            label = img.get('alt')
            src = img.get('src')

            meme_templates.append({'label': label, 'link': link, 'src': src})
    except ConnectionError as e:
        print(e)
    """
    meme_templates = []
    for link in links:
        meme_templates.append({'link': link})
    return meme_templates


def chrome_webpage(url):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    # this will disable image loading
    chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    # or alternatively we can set direct preference:
    chrome_options.add_experimental_option(
        "prefs", {"profile.managed_default_content_settings.images": 2}
    )
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Fetch the page
    driver.get(url)
    delay = 20
    try:
        WebDriverWait(driver, delay).until(
            EC.presence_of_element_located(
                (By.XPATH,
                 "//li[contains(@class, 'content-stream-item') and @path='-30902635.0.0']")
            )
        )
        print(f"{url} is ready!")
    except TimeoutException:
        print(f"{url} Loading took too much time!")


    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    SCROLL_PAUSE_TIME = 5
    MAX_SCOLL = 40
    count_scroll = 0
    number_scroll_trying = 0
    MAX_SCROLL_TRYING = 10
    while True:
        # Scroll down to bottom
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        try:
            new_height = driver.execute_script("return document.body.scrollHeight")
        except Exception as e:
            print(f"An error occurred: {e}")
            break

        if new_height == last_height:
            print(f"Max Height Reached? Attempt: {number_scroll_trying + 1}")
            if number_scroll_trying < MAX_SCROLL_TRYING:
                number_scroll_trying += 1
            else:
                print("Maximum height reached after multiple attempts.")
                break
        else:
            number_scroll_trying = 0
        driver.delete_all_cookies()
        if count_scroll >= MAX_SCOLL:
            break
        last_height = new_height
        print(f"Scroll Count: {count_scroll}")
        count_scroll += 1
        # Get the rendered HTML content
        html_content = driver.page_source

    return html_content


def crawl_template_page(temp):
    """Crawls data from the template page.

    Args:
        template_link (str): link identifier of the template
        page (int): page number
        num_retries (int): number of retries
    """

    url = temp["link"]
    html_path = temp["html_path"]
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    tree = html.fromstring(html_content)
    # XPath expression to find elements with the specified path attribute
    value = '30902635.0.0'
    xpath_query = f'//*[contains(@*, "{value}")]'
    memes = []

    path_value = '-30902635.0.0'
    all_elements = tree.xpath(f'//*[@path="{path_value}"]')
    for element in all_elements:
        # Extract Instance ID
        href_value = element.xpath(
            './/a[@class="comp-ui-link clickable2 instance"]/@href')[0]

        # Extract Text
        xpath_query = './/div[@class="text"]'
        text_string = [html_text.text_content().strip()
                       for html_text in element.xpath(xpath_query)]

        # Extract Image
        img_src = element.xpath('.//img/@src')[0]
        memes.append({"url": url, "instance_id": href_value,
                      "img_src": img_src, "text": text_string})
    print("Found memes: {}".format(len(memes)))
    return memes


def download_htmls(templates, templates_file):
    if not os.path.exists(HTML_PATH):
        os.makedirs(HTML_PATH)

    for index, template in enumerate(templates):
        url = template['link']
        name = url.split("/")[-1]
        output_file_path = HTML_PATH + f'/{name}.txt'
        templates[index]["html_path"] = output_file_path
        # Check if the HTML file already exists
        if os.path.exists(output_file_path):
            print(f"File already exists: {output_file_path}")
            continue

        # Crawl the HTML Files
        html_content = chrome_webpage(url)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(html_content)
        print(f"HTML content downloaded and saved: {output_file_path}")
        # save template information and load image
        templates_file.write(f'{url}\t{name}\n')
    templates_file.close()
    return templates


class MemeGeneratorCrawler:
    """MemeGenerator.net website crawler."""

    # characteristics of the website
    temp_pp = 15  # templates per page
    capt_pp = 15  # captions per page

    def __init__(self, poolsize=2,
                 min_len=10, max_len=96, max_tokens=31,
                 detect_english=False, detect_duplicates=False):
        """Initializes crawler and multiprocessing Pool.

        Args:
            poolsize (int): size of the multiprocessing pool
            min_len (int): minimum length of the caption text
            max_len (int): maximum length of the caption text
            max_tokens (int): maximum number of tokens in the caption text
            detect_english (bool): (non-stable) globally filter non-english templates
            detect_duplicates (bool): (slow) check for the similarity of captions and filter duplicates
        """

        # text preprocessing parameters
        self.min_len = min_len
        self.max_len = max_len
        self.max_tokens = max_tokens
        self.detect_english = detect_english
        self.detect_duplicates = detect_duplicates

        # containers shared across threads
        self.captions = {}
        self.num_visited = {}
        self.total_texts = {}

    def template_page_callback(self, result):
        """Processes the results from the template page."""
        _, memes, link = result

        # check and clear memes
        memes_filtered = []

        for meme in memes:
            (score, top, bottom) = meme
            top, bottom = clean_text(top), clean_text(bottom)
            text = (top + ' ' + bottom).lower()

            if check_text(text, min_len=self.min_len, max_len=self.max_len, max_tokens=self.max_tokens):
                memes_filtered.append((score, top, bottom))
                self.total_texts[link] += text + ' '

        self.captions[link] += memes_filtered
        self.num_visited[link] += 1

    def crawl_dataset(self, num_templates=300, num_captions=3000, save_dir='dataset/memes'):
        """Crawls dataset from memegenerator.net website.

        Args:
            num_templates (int): number of meme templates to crawl
            num_captions (int): number of captions per template
            save_dir (str): directory for saving the data
        """

        # directories and files
        images_dir = os.path.join(save_dir, "images/")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        templates_file = open(os.path.join(save_dir, "templates.txt"), 'a')
        captions_file = open(os.path.join(save_dir, "captions.txt"), 'a')

        # counters
        temp_page = 1
        total_captions = 0

        # start crawling until enough templates are loaded
        start_time = time.time()

        # parse page with templates
        templates = crawl_templates(page=temp_page)

        # download all templates
        templates = download_htmls(templates, templates_file)

        print(f'{time_to_str(time.time() - start_time)}, '
              f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
              f'Crawling page {temp_page} with {len(templates)} templates')

        memes = []
        for temp in templates:
            memes += crawl_template_page(temp)
            time.sleep(0.3)

        # das
        all_captions = []
        for meme in memes:

            """
            if self.detect_english:
                # check captions language
                prob_en = np.mean([english_prob(text) for _ in range(5)])
                if prob_en < 0.9:
                    # non-english, stop processing
                    print(f'{time_to_str(time.time() - start_time)}, '
                            f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
                            f'   NON_ENGLISH {label} - {len(self.captions[link])} captions (eng:{prob_en:.3f})')
                    continue
            else:
                prob_en = None
            """

            # ToDo: Remove Duplicates
            """
            if self.detect_duplicates:
                # check duplicates and keep collecting to get `n_captions_per_template`
                unique_captions = []
                while True:

                    # process crawled captions for duplicates (slow..)
                    for (score, top, bottom) in self.captions[link]:
                        is_unique = True
                        text = (top + ' ' + bottom).lower()

                        for (_, other_top, other_bottom) in unique_captions:
                            other_text = (other_top + ' ' +
                                            other_bottom).lower()
                            if sim_ratio(text, other_text) > 0.9:
                                is_unique = False
                                break

                        if is_unique:
                            unique_captions.append((score, top, bottom))

                    self.captions[link] = []
                    if len(unique_captions) >= num_captions:
                        break

                    # load five more pages
                    for i in range(page + 1, page + 10):
                        crawl_template_page(link)
                    page = i
            else:
                unique_captions = self.captions[link]
            """
            # ToDo: Check scoring

            """
            # take top captions by their score
            captions = list(sorted(unique_captions, key=lambda x: -x[0]))
            captions = captions[:num_captions]
            """
            link = meme["url"]
            instance_id = meme["instance_id"]
            src = meme["img_src"]
            captions = meme["text"]
            image_path = load_image(src, images_dir)

            # save captions
            top = captions[0]
            bot = captions[1]
            top, bot = clean_text(top), clean_text(bot)
            text = (top + ' ' + bot).lower()
            if not check_text(text, min_len=self.min_len,
                              max_len=self.max_len,
                              max_tokens=self.max_tokens):
                continue
            text = top + ' ' + SPECIAL_TOKENS['SEP'] + ' ' + bot
            all_captions.append(f'{link}\t{instance_id}\t{text}\t{src}\t{image_path}\n')

            """
            print(f'{time_to_str(time.time() - start_time)}, '
                    f'{100 * float(total_captions) / num_templates / num_captions:5.2f}%: '
                    f'   {label} - {len(captions)} captions ({total_captions}) (pid:{page}, en:{prob_en:.3f})')
            """

        time.sleep(0.5)
        all_captions = list(set(all_captions))
        print("Number of captions: {}".format(len(all_captions)))
        captions_file.write("".join(all_captions))

        print(f'Finished: crawled {len(templates)} templates')

        captions_file.close()
