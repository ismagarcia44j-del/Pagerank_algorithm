from pagerank import transition_model, sample_pagerank,iterate_pagerank
def test_transition_model():
    corpus = {
    "1.html": {"2.html"},
    "2.html": {"1.html", "3.html", "4.html"},
    "3.html": {"2.html", "4.html"},
    "4.html": {"2.html"},
    }

    page = "2.html"

    assert transition_model(corpus, "2.html", 0.85) == {
            "1.html": 0.3208333333333333,
            "2.html": 0.037500000000000006,
            "3.html": 0.3208333333333333,
            "4.html": 0.3208333333333333,
        }

def test_sample_pagerank():
    corpus = {
        "1.html": {"2.html"},
        "2.html": {"1.html", "3.html", "4.html"},
        "3.html": {"2.html", "4.html"},
        "4.html": {"2.html"},
        }
    result = sample_pagerank(corpus, 0.85, 10000)

    # Always true, regardless of randomness
    assert abs(sum(result.values()) - 1.0) < 1e-9
    assert set(result.keys()) == set(corpus.keys())

    # Page 2 should be the highest (it's the most linked-to)
    assert result["2.html"] > result["1.html"]
    assert result["2.html"] > result["3.html"]

    # By symmetry, Page 1 and Page 3 should be roughly equal
    assert abs(result["1.html"] - result["3.html"]) < 0.03

def test_iterate_pagerank():
    corpus = {
        "1.html": {"2.html"},
        "2.html": {"1.html", "3.html", "4.html"},
        "3.html": {"2.html", "4.html"},
        "4.html": {"2.html"},
        }
    damping_factor = 0.85
    it_pr = iterate_pagerank(corpus, damping_factor)
    assert sum(it_pr.values()) == 1