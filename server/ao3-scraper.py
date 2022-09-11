from pyquery import PyQuery as pq
from urllib.error import HTTPError
from time import sleep
import re

FILENAME = 'ao3'
OVERWRITE_FILE = False
PAGE = 1


def readChapters(url, chapters):
    try:
        for i, id in enumerate(chapters[1:]):
            print(f'- Chapter {i+2}')
            churl = f"{url}/chapters/{id}"
            chapter = pq(churl)

            with open('ao3.txt', 'a', encoding="utf-8") as file:
                chapter('.module p').each(lambda i, e: file.write(stripTags(e)))
            sleep(4)
    except HTTPError as err:
        # Probably got rate limited, wait a while before resuming.
        print(f'{err.code} Error: Waiting 2 minutes')
        sleep(120)

def stripTags(e):
    html = pq(e).html()
    html = re.sub('<[^>]+>', '', html)
    html = html.strip()
    html = re.sub(r'\n\s*\n','\n',html,re.MULTILINE)
    return f'{html}\n'

def getChapters(e):
    try:
        print(f"Reading '{pq(e).find('h4 a').eq(0).html()}' ({pq(e).find('h4 a').eq(1).html()})")
        url = f"https://archiveofourown.org{pq(e).find('h4 a').attr('href')}"
        main = pq(url)

        if main('#selected_id'):
            print('- Chapter 1')
            with open('ao3.txt', 'a', encoding='utf-8') as file:
                main('.userstuff + .module p').each(lambda i, e: file.write(stripTags(e)))

            chapters = []
            main('p #selected_id option').each(lambda i, e: chapters.append(pq(e).val()))
            readChapters(url, chapters)

        else:
            with open('ao3.txt', 'a', encoding="utf-8") as file:
                main('#chapters p').each(lambda i, e: file.write(stripTags(e)))
    except HTTPError as err:
        print(f'{err.code} Error: Waiting 2 minutes')
        sleep(120)


search = pq(f"https://archiveofourown.org/works/search?commit=Search&page={PAGE}&work_search%5Blanguage_id%5D=en&work_search%5Brating_ids%5D=13&work_search%5Bsingle_chapter%5D=1&work_search%5Bsort_column%5D=_score&work_search%5Bsort_direction%5D=desc")  # Any work list url

if OVERWRITE_FILE:
    with open(f'{FILENAME}.txt', 'w'): pass

search('li + .work').each(lambda i, e: getChapters(e))
