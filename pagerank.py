import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probab_distribution = dict()
    if corpus[page]:  # if page has links inside
        corpus_links_odd = (1-damping_factor)/len(corpus)  # overall odd for each link in corpus
        # odd for links inside the page # it added corpus_links_odd because all corpus values pages are in corpus key pages
        page_links_odd = (damping_factor/len(corpus[page])) + corpus_links_odd
        for link in corpus:
            if link in corpus[page]:  # if link belongs
                probab_distribution[link] = page_links_odd
            else:
                probab_distribution[link] = corpus_links_odd
    else:  # if doesn't have ...
        for link in corpus:
            probab_distribution[link] = 1/(len(corpus))
    return probab_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Choose randomly a page and save it in samples
    samples = [random.choice(list(corpus.keys()))]
    page_rank = dict()
    # Yield transition model for the next sample
    while len(samples) < n:
        probab_distribution = transition_model(corpus, samples[-1], damping_factor)
        pages = list(probab_distribution.keys())
        weights = list(probab_distribution.values())
        selected_page = str(random.choices(pages, weights, k=1)[0])
        samples.append(selected_page)

    for page in corpus:
        page_rank[page] = samples.count(page)/n

    return page_rank
def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Assigning default probability to each page
    new_pagerank = dict()
    for key in corpus:
        new_pagerank[key] = 1/len(corpus)
    # Defining the numerator of PR(p)
    numerator = 0
    difference = True

    # Iterate until difference of new-last are <=0.001
    while difference:
        last_pagerank = copy.deepcopy(new_pagerank)
        for p in list(corpus):  # PR(p)
            # check if any page (i) in corpus is linking to p
            for i, links in corpus.items():
                # corpus is true or all pages has the same P in i (is empty)
                links = corpus[i] or set(corpus)
                if p in links:  # is p any page of i's links?
                    numerator += last_pagerank[i]/len(links)
                    # numerator = PR(i) / Numlinks(i)
            # Computing PR(p)
            new_pagerank[p] = ((1 - damping_factor)/len(corpus))+(damping_factor * numerator)
            numerator = 0
        difference = any(abs(new-last) > 0.001 for last,
                         new in zip(last_pagerank.values(), new_pagerank.values()))
    return new_pagerank


if __name__ == "__main__":
    main()
