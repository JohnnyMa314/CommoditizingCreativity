import requests
import re
import os
import sys
import codecs
from bs4 import BeautifulSoup
from selenium import webdriver

def get_all_movies():
    '''
    Scrape 'http://www.imsdb.com/all%20scripts/' to extract the list of
    available scripts on IMSDb and the URL at which to access them.
    Returns:
    --------
    movie list: list of tuples
        each tuple contains:
        (movie title, link to movie page, movie_title)
        movie page: string with space and commas
        link: string href= ...
        movie_title: title with the whitespaces as _, name cropped at '.' or ','
    '''

    # Parse the page http://www.imsdb.com/all%20scripts/ with beautiful soup
    link_all_scripts = 'http://www.imsdb.com/all%20scripts/'
    response_all_scripts = requests.get(link_all_scripts)
    soup = BeautifulSoup(response_all_scripts.text, 'html.parser')

    # This webpage is constructed with tables, the 3rd one is the one we want
    find_tables = soup.findAll('td', valign='top')
    all_movies = find_tables[2].findAll('a')

    # Build the final list of tuples, which is to be returned
    movies = [(movie_info.string, \
              movie_info["href"], \
              re.split("[,.]",movie_info.string)[0].replace(' ', '_'))
              for movie_info in all_movies]
    return movies

def check_movie_info(movies):
    '''
    Check that the list of tuples (movie title, link, movie_title)
    in movies have a link that start with '/Movie Scripts/'
    Parameter
    ---------
    movies: list of tuples
        A list returned by the function `get_all_movies`
    Returns
    -------
    A string that indicates whether there was a problem or not
    '''
    for movie in movies:
        if movie[1][0:15] !='/Movie Scripts/':
            return 'One of the movie link does not start with /Movie Scripts/.'
    return 'All movie URLs have a correct format.'

def handle_movie (movie, browser):
    '''
    Download the script corresponding to `movie`, using selenium
    Parameters
    ----------
    movie: tuple
        a tuple from the `movies` list created by `get_all_movies`
            (movie title, link to movie page, movie_title)
    browser: object
        the browser used by selenium to get complete html page
    '''
    # Unpack tuple
    title, link_to_movie_page, movie_title = movie

    # Interrogate the page with all the movie information (ratings, writer,
    # genre, link to script)
    full_html_link = u'http://www.imsdb.com' + link_to_movie_page
    response_script = requests.get(full_html_link)
    soup = BeautifulSoup(response_script.text, 'html.parser')

    # Get all relevant information (genre, writer, script) from page
    list_links = soup.findAll('table', "script-details")[0].findAll('a')
    genre = []
    writer = []
    script = ''
    for link in list_links:
        href = link['href']
        if href[0:7]== "/writer":
            writer.append(link.get_text())
        if href[0:7]== "/genre/":
            genre.append(link.get_text())
        if href[0:9]== "/scripts/":
            script = href

    # If the link to the script points to a PDF, skip this movie, but log
    # the information in `movies_pdf_script.csv`
    if script == '' or script[-5:] != '.html':
        path_to_directory = './data/IMSDB/'
        pdf_logging_filename = path_to_directory + 'movies_pdf_script.csv'
        with open(pdf_logging_filename, 'a') as f:
            new_row = title + '\n'
            f.write(new_row)

    # If the link to the script points to an html page, write the corresponding
    # text to a file and include the movie in a csv file, with meta-information
    else:

        # Parse the webpage which contains the script text
        full_script_url =  u'http://www.imsdb.com' + script
        browser.get(full_script_url)
        page_text = browser.page_source
        soup = BeautifulSoup(page_text, 'html.parser')

        # If the scraping does not go as planned (unexpected structure),
        # log the file name in an error file
        if len(soup.findAll('td', "scrtext"))!=1:
            error_file_name = './data/IMSDB/scraping_error.csv'
            with open(error_file_name, 'a') as error_file:
                new_row = title + '\n'
                error_file.write( new_row )

        # Normal scraping:
        else:
            # Write the script text to a file
            path_to_directory = './data/IMSDB/texts/'
            filename = path_to_directory + movie_title + '.txt'
            text = soup.findAll('td', "scrtext")[0].get_text()
            with codecs.open(filename, "w",
                    encoding='ascii', errors='ignore') as f:
                f.write(text)

            # Add the meta-information to a CSV file
            path_to_directory = './data/IMSDB/'
            success_filename = path_to_directory + 'successful_files.csv'
            new_row = title + ',' + str(genre) + ',' + str(writer) + ',' \
                    + movie_title + ',' + filename + '\n'
            with open(success_filename, 'a') as f:
                f.write(new_row)

if __name__ == '__main__':

    CHROME_DRIVER_PATH = "./chromedriver-1"

    # Create data/scraping/texts files
    if not os.path.exists('./data'):
        os.mkdir('./data')
        print('making ./data folder')
    if not os.path.exists('./data/IMSDB'):
        os.mkdir('./data/IMSDB')
        print('making ./data/IMSDB folder')
    if not os.path.exists('./data/IMSDB/texts'):
        os.mkdir('./data/IMSDB/texts')
        print('making ./data/IMSDB/texts folder')

    # List all the available movies, and the corresponding URL links
    movies = get_all_movies()
    print(check_movie_info(movies))

    # Write all the scripts (in texts folder) and the summary of the movies
    # in .csv format (in scraping folder)
    browser = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH)
    for i,movie in enumerate(movies):
        handle_movie(movie, browser)
        print("----------------------")
        print(movie)
        print("----------------------")
        print("\n")