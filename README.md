## Commoditizing Creativity: DS 1001 Project

**Team Members:**
- Brian Pennisi (bp2221)
- Ted Xie (tx607)
- Amir Malek (am6370)
- Johnny Ma (jlm10003)

Film and television production companies have always followed a fixed sequence for content creation and business development. Broadly explained, this content creation lifecycle consists of a creative pitch and business negotiations, followed by pre-production, production, and post-production. Once these stages are complete, a film or show is ready for launch. Due to industry-wide interest in accelerating this content creation lifecycle, we decided to take a closer look at the business negotiations (more specifically, film financing), where a screenplay submitted for review is approved and financed for production. Studios receive thousands of solicited and unsolicited screenplays each day, and must hire and train professional script readers to perform “script coverage,” where scripts receive feedback as to their quality and expected cost of production. As the volume of received scripts is simply too high for the script readers to cover, most are filtered from consideration without even being read. If we can develop a data science based pipeline to instantly estimate a script’s expected production cost, we can prevent script readers from wasting time reading scripts the studios cannot or will not financially support. 

Simiarly to resume filters, we create a pipeline which takes raw movie scripts as an input and provides an estimated budget category, either low or high budget, as an output. From a business perspective, this would alleviate the creative bottleneck created by limited script readers, and allow for faster filtering of scripts from a large volume to meet budget requirements. For a small production lab such as NYU’s, where we estimate each reader spends around 6 hours per week reading scripts, the ability to automatically screen high budget scripts could allow them to identify lower budget scripts that the production company can actually produce.

We scrape 1,084 screenplays from [IMSDB.com](https://www.imsdb.com), an internet movie script database commonly used by industry and academics alike. The scripts contain scene descriptions, stage direction, and character dialogue, with an average of 5,000 lines of text and 24,000 tokens after preprocessing per screenplay. The observed budget for each film was collected from [boxofficemojo.com](https://www.boxofficemojo.com) and [the-numbers.com](https://www.the-numbers.com), the industry standard box office data websites for tracking movie financials. These budget values were obtained using the associated IMDB-ID (a unique identifier for each movie), as well as a name + year search using fuzzy string matching, with low probability matches manually fixed for accuracy.

We use TF-IDF, off the shelf Named Entity Recognition using Spacy, and averaged word embeddings using Word2Vec. We use Logistic Regression with L2 regularization, Gradient Boosting Machine, and Support Vector Machine for budget classification. The results of our combined features on the best performing classifier, Logistic Regression, are shown below. 

|      Features      | Precision | Recall | Accuracy | F1-Score |
|:------------------:|:---------:|:------:|:--------:|:--------:|
|       TF-IDF       |    0.71   |  0.71  |   0.711  |   0.712  |
|    **TF-IDF + NER**    |    **0.72**   |  **0.72**  |   **0.723**  |   **0.724**  |
|    TF-IDF + W2V    |    0.68   |  0.69  |   0.684  |   0.684  |
|      NER + W2V     |    0.64   |  0.64  |   0.636  |   0.636  |
| TF-IDF + NER + W2V |    0.69   |  0.69  |   0.690  |   0.690  |

![Budget Distribution](/figures/Film_Budget.png)
![ROC curve for best model](/figures/AUC.png)