import time

import ftfy
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import process
from tqdm import tqdm


def main():
    feats = ['score', 'date', 'title', 'budget', 'dom', 'intl']
    tnums = pd.DataFrame(columns=feats)

    # get budgets from list of budgets on The Numbers.com
    pages = range(1, 6101, 100)
    for page in pages:
        time.sleep(1)
        if page == '1':
            url = 'https://www.the-numbers.com/movie/budgets/all'
        else:
            url = 'https://www.the-numbers.com/movie/budgets/all/' + str(page)
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table')

        # get table of financials
        for row in table.findAll("tr")[1:]:
            tags = []
            for tag in row:
                if tag != '\n':
                    tags.append(ftfy.fix_text(tag.text.strip()))  # clean up unicode
            tnums = tnums.append(pd.DataFrame([tags], columns=feats))

    # output
    tnums = tnums.set_index('score')
    tnums.to_csv('./data/TN/movie_financials.csv')

    ## merging financial information with budget via Title and Date
    df = pd.read_csv("./data/IMSDB/movie_info.csv", index_col=0)
    df = df[~df['Release-Date'].isna()]

    years = []
    for date in tnums['date']:
        years.append(date[-4:])
    tnums['Title'] = tnums['title'] + ' (' + years + ')'

    years = []
    for date in df['Release-Date']:
        years.append(date[-4:])

    df['Title'] = df['Title'] + ' (' + years + ')'

    df2 = pd.DataFrame(columns=['ind1', 'ind2'])
    for film in tqdm(df['Title']):
        match = process.extractOne(film, tnums['Title'])
        print(film)
        print(match)
        print('\n')

        if match[1] >= 90:  # if title is likely to be matched
            ind = tnums.Title[tnums.Title == match[0]].index
            print(ind.values)
            if ind.any():
                print('yes')
                df2 = df2.append({'ind1': df[df['Title'] == film].index.item(), 'ind2': ind.item()}, ignore_index=True)

    df2.to_csv('IMDB_TN_XC.csv')
    df2 = df2.set_index('ind1')

    merged_TN = df2.merge(tnums, how='left', left_on='ind2', right_index=True)
    merged_TN['Budget'] = [bud.strip('$') for bud in merged_TN['budget']]
    merged_IMDB_TN = df.merge(merged_TN['Budget'], how='left', left_index=True, right_index=True)

    # replacing missing BOM budgets with TN budgets
    for ind, row in enumerate(merged_IMDB_TN['Budget_x']):
        if str(row) == 'nan':
            print(row)
            merged_IMDB_TN.loc[ind, 'Budget'] = merged_IMDB_TN.loc[ind, 'Budget_y']
        else:
            merged_IMDB_TN.loc[ind, 'Budget'] = merged_IMDB_TN.loc[ind, 'Budget_x']

    movie_info_out = merged_IMDB_TN.drop(labels=['Budget_x', 'Budget_y'], axis=1)
    movie_info_out.to_csv('final_movie_info.csv')
    missing_budget = movie_info_out[movie_info_out['Budget'].isna()]
    missing_budget.to_csv('missing_budget.csv')


if __name__ == '__main__':
    main()
