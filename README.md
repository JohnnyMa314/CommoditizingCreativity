## Commoditizing Creativity: DS 1001 Project

**Team Members:**
- Brian Pennisi (bp2221)
- Ted Xie (tx607)
- Amir Malek (am6370)
- Johnny Ma (jlm10003)

Can you predict the success of a movie just from the text of its screenplay? Hundreds of script readers and development interns at major creative firms are tasked with this exact prediction problem. Every year, thousands of scripts are submitted for production consideration, but only dozens are greenlit and even fewer are profitable. If we can build a model to predict a script’s expected popularity or financial return, we can not only remove the cost of hiring and training script readers, but also understand what makes a script a blockbuster and reduce the risk of box office bombs at the writing level. 


The supervised learning problem is as follows: given the textual features of a screenplay, predict the gross revenue. We use revenue, budget, and other release data like # screens or date of release from boxofficemojo.com which has an API. We scrape screenplays from IMSDB (code ready), use a research ready film corpus, or use subtitle data. We can link film data through IMDB ids or fuzzy string matches on film names and year. We can start from simple tools like BoW or sentiment analysis feature extraction and scale up to more complex LMs or engineered features like narrative structure, narrative events, or dialogue networks. We can also include metadata like genre, production company, stars, etc. using IMDB identifiers. We can evaluate our model using standard prediction measures, with an emphasis on precision in predicting placement in bucketed categories, from duds to mega hits. 


Alternative supervised learning problems are as follows: given the textual features of a screenplay, predict the binary decision whether or not to acquire and produce the script. This problem statement is closer to the actual job of script readers. However, it is difficult to find unproduced scripts due to selection bias. We can use community gathered script hubs (database or industry curated) which contain unproduced screenplays, and assume the ones not yet made will never be made. Ideally, we can work with NYU Production Lab, which receives many scripts but funds only a few, to get their data in return for building a model to help them distribute limited grant money. Could even turn into a Capstone project. We would appreciate any advice on convincing the Production Lab to enter into this mutually beneficial arrangement. 


If either of these approaches don’t work, we can fallback on the supervised problem of predicting a script’s budget given the text, which would also be financially valuable information. If we are able to build even a mildly successful script success prediction engine, this technology would be a big step towards revolutionizing creative industries with AI writing assistance, towards the final goal of full automated entertainment content. 


## Tasks

- [x] Literature Review
	- [x] Script Papers
	- [ ] Genre Classification

- [x] Scrape Scripts
- [ ] Get IMDB Ratings Data
- [x] Get Box Office Revenue Data
- [x] Merge Data Source together

- [x] Build NLP Feature Pipeline
	- [x] Bag of Words
	- [ ] Word2vec
	- [ ] Transformers

