import os
import time
from lxml import html
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
LINKS = [
    # Religion
    "https://memegenerator.net/Advicejew",
    "https://memegenerator.net/Confused-Muslim-Girl",
    "https://memegenerator.net/Angry-Muslim-Guy",
    "https://memegenerator.net/Scumbag-God",
    "https://memegenerator.net/Ordinary-Muslim-Man",
    "https://memegenerator.net/Scumbag-Catholic-Priest",

    "https://memegenerator.net/Condescending-Christian",
    "https://memegenerator.net/Asshole-Christian-Missionary",
    "https://memegenerator.net/Typical-Atheist",
    "https://memegenerator.net/Uncontested-Atheist",
    "https://memegenerator.net/Buddha",
    "https://memegenerator.net/Buddhadawg",


    # Sexual Orientation
    "https://memegenerator.net/Gay-Richard-Simmons",
    "https://memegenerator.net/Chinese-Lesbians",
    "https://memegenerator.net/Transvestite-Trevor",

    # Race/Nationality
    "https://memegenerator.net/Asian-College-Freshman",
    "https://memegenerator.net/Sassy-Black-Woman",
    "https://memegenerator.net/Scumbag-Whitehouse",
    "https://memegenerator.net/Scumbag-America",
    "https://memegenerator.net/Stereotypical-Redneck",
    "https://memegenerator.net/Stereotypical-Indian-Telemarketer",
    "https://memegenerator.net/Canada-Flag",
    "https://memegenerator.net/Average-Italian-Driver",
    "https://memegenerator.net/Average-Italian-Criminal",
    "https://memegenerator.net/Success-Germany",
    "https://memegenerator.net/Germany-Pls",
    "https://memegenerator.net/Indian-Father",

    "https://memegenerator.net/Typical-Germany-Lover",
    "https://memegenerator.net/Average-Italian-Guy-Official",
    "https://memegenerator.net/Frenchy",
    "https://memegenerator.net/Scumbag-French",
    "https://memegenerator.net/Stereotypical-French-Man",
    "https://memegenerator.net/Advice-Kpop-Fangirl",
    #"https://memegenerator.net/American-Pride-Eagle",

    "https://memegenerator.net/Good-Chinese-Student",
    "https://memegenerator.net/Stern-But-Honest-Chinese-Guy",

    # Socio
    "https://memegenerator.net/Homeless-Man-2",

    # Gender
    "https://memegenerator.net/Pms-Woman",
    "https://memegenerator.net/3-Lesbians-Showing-And-Fingering-Their-Ass",
    "https://memegenerator.net/Lesbian-Scissor",
    "https://memegenerator.net/Chinese-Lesbians",
    "https://memegenerator.net/Gay-Pornstar-Logic",
    "https://memegenerator.net/Gay-Pride-Queer",
    "https://memegenerator.net/Oppressive-Trans-Bro",
    "https://memegenerator.net/Transvestite-Trevor",

    # Politics
    "https://memegenerator.net/Feminist-Cunt",
    "https://memegenerator.net/Feministfrequently",
    "https://memegenerator.net/Slavery",
    "https://memegenerator.net/12-Years-A-Slave-Hangover",
    "https://memegenerator.net/Privilege-Abusing-White-Couple",
    "https://memegenerator.net/Muslim-Immigrant",
    "https://memegenerator.net/Feminist-Cunt",
    "https://memegenerator.net/Scumbag-Police-Officer",
    "https://memegenerator.net/Strict-Policeman",
    "https://memegenerator.net/Will-Sons-Of-Guns",
    "https://memegenerator.net/Native-American",
    "https://memegenerator.net/Privilege-Denying-Dude",
    "https://memegenerator.net/Privilege-Denying-Tranny",


    "https://memegenerator.net/Jesus-Christ",
    "https://memegenerator.net/Ordinary-Muslim-Man",
    "https://memegenerator.net/Confused-Muslim-Girl",
    "https://memegenerator.net/Jewish-Dude",
    "https://memegenerator.net/Like-A-Jew",
    "https://memegenerator.net/American-Flag-Shotgun-Guy",
    "https://memegenerator.net/Obese-American",
    "https://memegenerator.net/Mexicanotriste",
    "https://memegenerator.net/Mexicans-On-A-Truck",
    "https://memegenerator.net/Nia-China",
    "https://memegenerator.net/Feministfrequently",
    "https://memegenerator.net/Privilege-Denying-Feminist",


    # Body Type
    "https://memegenerator.net/Skinny-Kid",
    "https://memegenerator.net/Fat-Girl",


    "https://memegenerator.net/Bad-Advice-Asian",
    "https://memegenerator.net/Troll-Asian",

    "https://memegenerator.net/Y-U-So-Arab",
    "https://memegenerator.net/Arabic-Meme",
    "https://memegenerator.net/Richarabclap",

    "https://memegenerator.net/Ignorant-White-Girl",
    "https://memegenerator.net/Nice-White-Girl",
    "https://memegenerator.net/White-Power-Problems",
    "https://memegenerator.net/Policeman",

    ]


def crawl_templates(page=1):
    """Crawls templates from All-time page.

    Args:
        page (int): page number
    """

    meme_templates = []

    meme_templates = []
    for link in LINKS:
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
    MAX_SCOLL = 200
    count_scroll = 0
    number_scroll_trying = 0
    MAX_SCROLL_TRYING = 15
    while True:
        # Scroll down to bottom
        try:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        try:
            new_height = driver.execute_script(
                "return document.body.scrollHeight")
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
            print(f"Scroll Count: {count_scroll}")
            count_scroll += 1

        driver.delete_all_cookies()
        if count_scroll >= MAX_SCOLL:
            break
        last_height = new_height

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
        name = url.split("/")[-1]
        templates_file.write(f'')
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
            print("For {}.\n".format(temp))
            time.sleep(0.3)

        # das
        all_captions = []
        for meme in memes:
            link = meme["url"]
            label_name = link.split("/")[-1]
            instance_id = meme["instance_id"]
            src = meme["img_src"]
            captions = meme["text"]

            if src == "/img/empty.png":
                continue

            image_path = load_image(label_name, src, images_dir)

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
            image_path = image_path.split("/")[-1].split(".")[0].split("_")[-1]
            link = label_name + "_" + image_path
            instance_id = instance_id.split("/")[-1]
            all_captions.append(
                f'{link}\t{instance_id}\t{text}\n')

        time.sleep(0.5)
        all_captions = list(set(all_captions))
        print("Number of captions: {}".format(len(all_captions)))
        captions_file.write("".join(all_captions))

        print(f'Finished: crawled {len(templates)} templates')

        captions_file.close()
