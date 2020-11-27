import codecs
import datetime
import os
import re

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from imdb import IMDb
from selenium import webdriver


# get imdb id from title search, given higher than 2000 votes on movie
def get_imdb_from_title(movie_title):
    ia = IMDb()
    s_result = ia.search_movie(movie_title)

    if s_result is None:
        return '0'

    item = []
    # Print the long imdb canonical title and movieID of the results.
    for item in s_result[0:10]:
        print(item['long imdb canonical title'], item.movieID)
        movie = ia.get_movie(item.movieID)

        # if more than 10k people have seen the movie
        votes = movie.get('votes')
        if votes is not None:
            if votes > 2000:
                break
    if not item:
        return '0'

    return item.movieID


# get box office data manually from imdb id.
def get_boxoffice_data(imdb_id):
    # get url from
    url = 'http://www.boxofficemojo.com/title/tt' + str(imdb_id)
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    print(url)

    features = ['BOM-url', 'Distributor', 'OW-Revenue', 'OW-Opens', 'Budget', 'Release-Date', 'Days-in-Release', 'MPAA',
                'Runtime', 'Genres', 'Widest-Release']

    try:
        # get BOM_ID from BOM IMDB id page
        bom_id = 'N/A'
        out = soup.findAll('a', {'class': 'a-link-normal'})
        for tag in out:
            if '/release/' in tag.get('href'):
                bom_url = tag.get('href').replace('/release', '')
                bom_id = re.search('(?<=\/)(.*?)(?=\/)', bom_url).group()  # get bom "RL" id
                if 'rl' in bom_id:
                    break

        if bom_id == 'N/A':
            return pd.DataFrame('N/A', index=np.arange(1), columns=features)

        # fill out DataFrame
        bom_data = pd.DataFrame(columns=features)

        # get BOM meta data
        url = 'https://www.boxofficemojo.com/release/' + bom_id
        print(url)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        sum_table = soup.findAll('div', {'class': 'mojo-summary-values'})
        for info in sum_table:
            tags = info.findAll('div', {'class': 'a-section'})

            dist = tags[0].text.replace("Distributor", '').replace('See full company information\n\n', '')
            OW_revenue = tags[1].find('span', {'class': 'money'}).text.strip('$')
            OW_opens = tags[1].text.replace('Opening', '').replace('\n            theaters', '').replace(OW_revenue,
                                                                                                         '').strip('$')

            budget = tags[2].text.replace('Budget', '').strip('$')
            if 'Release Date' in budget:
                budget = 'N/A'
                tags[2:2] = ['N/A']

            temp = tags[3].text.replace('Release Date', '').replace('\n            -\n           ', '').strip()
            release_date = temp[:re.search('(19|20)\d{2}', temp).span()[1]]

            mpaa = tags[4].text.replace('MPAA', '')
            if 'Running Time' in mpaa:
                mpaa = 'N/A'
                tags[4:4] = ['N/A']

            # run time possibilities
            time_tag = tags[5].text.replace('Running Time', '')
            if 'min' in time_tag and 'hr' in time_tag:
                time = datetime.datetime.strptime(time_tag, '%H hr %M min').time()
                runtime = time.hour * 60 + time.minute
            if 'min' in time_tag and 'hr' not in time_tag:
                time = datetime.datetime.strptime(time_tag, '%M min').time()
                runtime = time.minute
            if 'min' not in time_tag and 'hr' in time_tag:
                time = datetime.datetime.strptime(time_tag, '%M min').time()
                runtime = time.hour * 60

            genres = tags[6].text.replace('Genres', '').replace('\n', '').split()
            days_in_release = tags[7].text.replace('In Release', '').split('/')[0]
            widest_release = tags[8].text.replace('Widest Release', '').replace('theaters', '').strip()

        # get domestic and total grosses in this time period
        gross_table = soup.findAll('div', {'class': 'mojo-performance-summary-table'})
        for gross in gross_table:
            grosses = gross.findAll('span', {'class': 'a-size-medium'})
            dom_gross = grosses[0].text.replace('\n', '').strip('$')
            int_gross = grosses[1].text.replace('\n', '').strip('$')
            WW_gross = grosses[2].text.replace('\n', '').strip('$')

        # outputting data
        bom_data = bom_data.append({'BOM-url': url,
                                    'Distributor': dist,
                                    'OW-Revenue': OW_revenue,
                                    'OW-Opens': OW_opens,
                                    'Domestic-Total': dom_gross,
                                    'WorldWide-Total': WW_gross,
                                    'Budget': budget,
                                    'Release-Date': release_date,
                                    'Days-in-Release': days_in_release,
                                    'MPAA': mpaa,
                                    'Runtime': runtime,
                                    'Genres': str(genres),
                                    'Widest-Release': widest_release}, ignore_index=True)

        return bom_data

    except:
        print('Error in reading Box Office Mojo Data')
        return pd.DataFrame('N/A', index=[0], columns=features)


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
               re.split("[,.]", movie_info.string)[0].replace(' ', '_'))
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
        if movie[1][0:15] != '/Movie Scripts/':
            return 'One of the movie link does not start with /Movie Scripts/.'
    return 'All movie URLs have a correct format.'


def handle_movie(movie, browser):
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
        if href[0:7] == "/writer":
            writer.append(link.get_text())
        if href[0:7] == "/genre/":
            genre.append(link.get_text())
        if href[0:9] == "/scripts/":
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
        full_script_url = u'http://www.imsdb.com' + script
        browser.get(full_script_url)
        page_text = browser.page_source
        soup = BeautifulSoup(page_text, 'html.parser')

        # If the scraping does not go as planned (unexpected structure),
        # log the file name in an error file
        if len(soup.findAll('td', "scrtext")) != 1:
            error_file_name = './data/IMSDB/scraping_error.csv'
            with open(error_file_name, 'a') as error_file:
                new_row = title + '\n'
                error_file.write(new_row)

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
            print(title)
            imdb_id = get_imdb_from_title(title)  # get imdb by searching text of title
            bom_data = get_boxoffice_data(imdb_id).iloc[0]  # get box office data from imdb
            imsdb_out = pd.DataFrame({'Title': title,
                                      'IMSDB-Genres': str(genre),
                                      'Writers': str(writer),
                                      'Movie-Title': movie_title,
                                      'IMDB-ID': imdb_id,
                                      'Filename': filename}, index=[0])

            df = pd.concat([imsdb_out, bom_data.to_frame().reindex().T], axis=1)  # concat the two dataframes
            return df


if __name__ == '__main__':

    CHROME_DRIVER_PATH = "./chromedriver"
    path_to_directory = './data/IMSDB/'

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
    big_df = pd.DataFrame()
    for i, movie in enumerate(movies):
        big_df = big_df.append(handle_movie(movie, browser), ignore_index=True)
        big_df.to_csv(path_to_directory + 'movie_info.csv')  # update csv iteratively
        print("----------------------")
        print(movie)
        print("----------------------")
        print("\n")
