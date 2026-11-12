#!/usr/bin/env python3
"""Resolve every reference against Crossref and write refs.bib plus a
verification log.  DOI entries are fetched directly; title-query entries are
accepted only when the returned title and first author match.  Abstracts are
cached (Crossref, then Europe PMC) for fact checking; they are not redistributed.
"""
import json, os, re, sys, time, html, difflib, unicodedata, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from refs_list import DOI, Q

CACHE = os.path.join(ROOT, "results", "crossref_cache")
os.makedirs(CACHE, exist_ok=True)
UA = "refcheck/0.1 (research manuscript reference verification)"

def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa
            err = e
            time.sleep(1.5 * (k + 1))
    print("FAIL", url[:110], repr(err)[:80], file=sys.stderr)
    return None

def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def by_doi(key, doi):
    f = os.path.join(CACHE, key + ".json")
    if os.path.exists(f):
        return json.load(open(f))
    d = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe=""))
    if d and d.get("status") == "ok":
        json.dump(d["message"], open(f, "w"))
        return d["message"]
    return None

def by_query(key, title, fam, year):
    f = os.path.join(CACHE, key + ".json")
    if os.path.exists(f):
        return json.load(open(f))
    q = urllib.parse.quote(f"{title} {fam} {year}")
    d = get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=6")
    if not d:
        return None
    best, bs = None, 0
    for it in d["message"]["items"]:
        t = (it.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, norm(t), norm(title)).ratio()
        fams = [norm(a.get("family", "")) for a in it.get("author", [])]
        okfam = any(norm(fam) in x or x in norm(fam) for x in fams if x)
        if it.get("type") not in ("journal-article", "book-chapter", "proceedings-article"):
            s -= 0.3
        if okfam and s > bs:
            best, bs = it, s
    if best is not None and bs >= 0.86:
        best["_match"] = round(bs, 3)
        json.dump(best, open(f, "w"))
        return best
    print(f"NOMATCH {key}: best={bs:.2f} {((best or {}).get('title') or [''])[0][:70]}", file=sys.stderr)
    return None

ABBR = {
 "Proceedings of the National Academy of Sciences": "Proc. Natl. Acad. Sci. USA",
 "Proceedings of the Royal Society B: Biological Sciences": "Proc. R. Soc. B",
 "Proceedings of the Royal Society of London. Series B: Biological Sciences": "Proc. R. Soc. B",
 "Evolution and Human Behavior": "Evol. Hum. Behav.",
 "Nature Communications": "Nat. Commun.", "Scientific Reports": "Sci. Rep.",
 "Science Advances": "Sci. Adv.", "Nature Ecology &amp; Evolution": "Nat. Ecol. Evol.",
 "Nature Ecology & Evolution": "Nat. Ecol. Evol.", "Nature Human Behaviour": "Nat. Hum. Behav.",
 "Journal of Theoretical Biology": "J. Theor. Biol.", "Animal Behaviour": "Anim. Behav.",
 "The American Naturalist": "Am. Nat.", "Human Nature": "Hum. Nat.",
 "Current Anthropology": "Curr. Anthropol.", "Behavioral and Brain Sciences": "Behav. Brain Sci.",
 "Journal of Personality and Social Psychology": "J. Pers. Soc. Psychol.",
 "Frontiers in Ecology and Evolution": "Front. Ecol. Evol.", "Frontiers in Psychology": "Front. Psychol.",
 "Evolutionary Anthropology: Issues, News, and Reviews": "Evol. Anthropol.",
 "Journal of Evolutionary Biology": "J. Evol. Biol.", "Journal of Human Evolution": "J. Hum. Evol.",
 "American Journal of Physical Anthropology": "Am. J. Phys. Anthropol.",
 "Behavioral Ecology": "Behav. Ecol.", "Behavioral Ecology and Sociobiology": "Behav. Ecol. Sociobiol.",
 "PLoS ONE": "PLoS ONE", "PLOS ONE": "PLoS ONE", "PLOS Biology": "PLoS Biol.",
 "PLOS Computational Biology": "PLoS Comput. Biol.", "Evolutionary Psychology": "Evol. Psychol.",
 "Psychological Science": "Psychol. Sci.", "Current Biology": "Curr. Biol.",
 "The Quarterly Review of Biology": "Q. Rev. Biol.", "American Anthropologist": "Am. Anthropol.",
 "Annual Review of Psychology": "Annu. Rev. Psychol.", "Behavior Genetics": "Behav. Genet.",
 "Personality and Individual Differences": "Pers. Individ. Differ.",
 "Journal of Archaeological Science": "J. Archaeol. Sci.", "Biology Letters": "Biol. Lett.",
 "Systematic Biology": "Syst. Biol.", "Genome Research": "Genome Res.",
 "Cross-Cultural Research": "Cross-Cult. Res.", "Psychological Inquiry": "Psychol. Inq.",
 "Review of General Psychology": "Rev. Gen. Psychol.", "JAMA Neurology": "JAMA Neurol.",
 "European Journal of Archaeology": "Eur. J. Archaeol.", "Biological Theory": "Biol. Theory",
 "Adaptive Behavior": "Adapt. Behav.", "Biology &amp; Philosophy": "Biol. Philos.",
 "Biology & Philosophy": "Biol. Philos.", "Perspectives in Biology and Medicine": "Perspect. Biol. Med.",
 "Personality and Social Psychology Bulletin": "Pers. Soc. Psychol. Bull.",
 "Personality and Social Psychology Review": "Pers. Soc. Psychol. Rev.",
 "Neuroscience &amp; Biobehavioral Reviews": "Neurosci. Biobehav. Rev.",
 "Neuroscience & Biobehavioral Reviews": "Neurosci. Biobehav. Rev.",
 "The Bell Journal of Economics": "Bell J. Econ.", "The American Mathematical Monthly": "Am. Math. Mon.",
 "Behavior Research Methods": "Behav. Res. Methods",
 "Journal of Experimental Psychology: General": "J. Exp. Psychol. Gen.",
 "Ecology and Evolution": "Ecol. Evol.", "Eugenics Quarterly": "Eugen. Q.",
 "American Journal of Biological Anthropology": "Am. J. Biol. Anthropol.",
 "Trends in Ecology &amp; Evolution": "Trends Ecol. Evol.", "Trends in Ecology & Evolution": "Trends Ecol. Evol.",
 "Comparative Biochemistry and Physiology Part A: Molecular &amp; Integrative Physiology": "Comp. Biochem. Physiol. A",
 "Comparative Biochemistry and Physiology Part A: Molecular & Integrative Physiology": "Comp. Biochem. Physiol. A",
 "Bulletin of the American Schools of Oriental Research": "Bull. Am. Sch. Orient. Res.",
 "Journal of Archaeological Method and Theory": "J. Archaeol. Method Theory",
 "Adaptive Human Behavior and Physiology": "Adapt. Hum. Behav. Physiol.",
 "The Innovation": "Innovation", "Human Biology": "Hum. Biol.",
}

def tex(s):
    s = unicodedata.normalize("NFC", html.unescape(s or ""))
    s = re.sub(r"<(i|em)>(.*?)</\1>", r"\\textit{\2}", s, flags=re.S)
    s = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", s)
    s = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "--").replace("\u2014", "---")
    s = s.replace("\u2018", "`").replace("\u2019", "'").replace("\u201c", "``").replace("\u201d", "''")
    s = s.replace("\u00a0", " ").replace("\u2009", " ")
    return s

def initials(given):
    parts = re.split(r"[\s]+", unicodedata.normalize("NFC", given or "").replace("\u2010", "-").replace(".", ". ").strip())
    out = []
    for p in parts:
        if not p:
            continue
        sub = [x for x in p.split("-") if x]
        out.append("-".join(x[0].upper() + "." for x in sub))
    return " ".join(out)

def authors(m):
    L = []
    for a in m.get("author", []):
        if "family" in a:
            fam = tex(a["family"])
            if fam.isupper() and len(fam) > 3:
                fam = fam.title()
            ini = initials(a.get("given", ""))
            L.append("{%s}, %s" % (fam, ini) if ini else "{%s}" % fam)
        elif "name" in a:
            L.append("{%s}" % tex(a["name"]))
    if len(L) > 8:
        L = L[:6] + ["others"]
    return " and ".join(L)

def year(m):
    for k in ("published-print", "issued", "published-online", "published"):
        dp = (m.get(k) or {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            return dp[0][0]
    return None

def sentence_case(t):
    """Titles deposited in Title Case are lower-cased except after a colon/period
    and for tokens that look like proper nouns already protected upstream."""
    return t

def entry(key, m, fix=None):
    fix = fix or {}
    typ = m.get("type")
    title = tex((m.get("title") or [""])[0])
    sub = m.get("subtitle") or []
    if sub and norm(sub[0]) and norm(sub[0]) not in norm(title):
        title = title.rstrip(":") + ": " + tex(sub[0])
    cont = html.unescape((m.get("container-title") or [""])[0])
    jr = ABBR.get(cont, ABBR.get(cont.replace("&", "&amp;"), tex(cont)))
    f = {"author": authors(m), "title": "{" + title + "}", "year": str(year(m))}
    if typ == "book-chapter":
        kind = "incollection"; f["booktitle"] = jr; f["publisher"] = tex(m.get("publisher", ""))
    else:
        kind = "article"; f["journal"] = jr
    if m.get("volume"): f["volume"] = m["volume"]
    if m.get("issue"): f["number"] = m["issue"].replace("/", "--")
    pg = m.get("page") or m.get("article-number")
    if pg: f["pages"] = pg.replace("-", "--").replace("----", "--")
    f["doi"] = m["DOI"]
    f.update(fix)
    body = ",\n".join("  %s = {%s}" % (k, v) for k, v in f.items() if v)
    return "@%s{%s,\n%s\n}\n" % (kind, key, body)

def abstract(m):
    a = m.get("abstract")
    if a:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(a))).strip()
    d = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%s&format=json&resultType=core"
            % urllib.parse.quote('"' + m["DOI"] + '"'), tries=2)
    try:
        r = d["resultList"]["result"][0]
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.get("abstractText", ""))).strip()
    except Exception:
        return ""

if __name__ == "__main__":
    want_abs = "--abstracts" in sys.argv
    recs, log = {}, []
    for key, doi in DOI.items():
        m = by_doi(key, doi)
        recs[key] = m
        log.append((key, "doi", doi, bool(m)))
    for key, (t, fam, yr) in Q.items():
        m = by_query(key, t, fam, yr)
        recs[key] = m
        log.append((key, "query", (m or {}).get("DOI", ""), bool(m)))
    missing = [k for k, m in recs.items() if not m]
    print("resolved %d / %d; missing: %s" % (len(recs) - len(missing), len(recs), missing))
    json.dump({k: {"doi": (m or {}).get("DOI"), "title": ((m or {}).get("title") or [""])[0],
                   "first_author": ((m or {}).get("author") or [{}])[0].get("family"),
                   "container": ((m or {}).get("container-title") or [""])[0],
                   "year": year(m) if m else None, "volume": (m or {}).get("volume"),
                   "page": (m or {}).get("page") or (m or {}).get("article-number"),
                   "match": (m or {}).get("_match")} for k, m in recs.items()},
              open(os.path.join(ROOT, "results", "ref_verification.json"), "w"), indent=1, ensure_ascii=False)
    if want_abs:
        absf = os.path.join(ROOT, "results", "abstracts_cache.json")
        A = json.load(open(absf)) if os.path.exists(absf) else {}
        for k, m in recs.items():
            if m and k not in A:
                A[k] = abstract(m)
                json.dump(A, open(absf, "w"), indent=0, ensure_ascii=False)
        print("abstracts:", sum(1 for v in A.values() if v), "of", len(A))


# ---------------------------------------------------------------------------
# BibTeX generation (run with --bib).  Titles deposited in Title Case or ALL CAPS
# are converted to sentence case; proper nouns are protected by a keep-list.
KEEP = """United States Poland Kenya Maasai Flores Philippines Ariaal Africa DRD4 Y D-PLACE Homo Neanderthal Neanderthals Neandertal Pleistocene Holocene Plio-Pleistocene Hadza Tsimane Ache Meriam Waorani
Ecuador Africa African Kenya Turkana Europe Eurasia China Xuchang Harbin Flores Indonesia Altai Bruniquel France Neolithic
Israel Lanchester's Bateman's Darwin Fisher Darwin-Fisher Darwin--Fisher Machiavellian Flynn Framingham British Swedish
Bolivian Amazon Amazonian Mesoamerica Mesoamerican Etlatongo Oaxaca Mexico America America's Ghazal Jordan Abşeron
Başur Höyük Mesopotamia Japanese WHRs BMIs DNA GWAS IQ MRI Sima Huesos Dinaledi Paleolithic Palaeolithic Mid-Pleistocene
Australopithecus Darwinism Bushmen Ju/'hoansi Amerindian Eurasia Zoon Politikon DeSilva Acheulean Wonderwerk Gesher Benot
Ya'aqov Liang Bua Dmanisi ARHGAP11B Tanzania Senegal Sereer Agta Philippines Finnish Iceland Icelandic Siberia Britain
Nordic Denmark Bateman""".split()
KEEPMAP = {w.lower(): w for w in KEEP}
LOWER_OK = {"the", "a", "an", "of", "in", "and", "or", "for", "to", "on", "by", "with", "from", "as", "at", "is", "are", "vs", "vs."}

def needs_case(t):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", t) if len(w) >= 4]
    if not words:
        return False
    cap = sum(1 for w in words if w[0].isupper())
    return t.isupper() or cap / len(words) > 0.6

def sentence(t):
    out, start = [], True
    for tok in re.split(r"(\s+)", t):
        if not tok.strip():
            out.append(tok); continue
        core = tok.strip("\"'`()[],;:?!.")
        low = core.lower()
        if tok.startswith("\\") or "{" in tok:
            new = tok
        elif low in KEEPMAP and not (start and False):
            new = tok.replace(core, KEEPMAP[low])
        elif start:
            new = tok.replace(core, core[:1].upper() + core[1:].lower()) if core else tok
        else:
            new = tok.replace(core, core.lower()) if core else tok
        out.append(new)
        start = tok.rstrip("\"'`)").endswith((":", "?", ".", "!")) and len(core) > 1
    return "".join(out)

AUTH = {
 "rowe1996": "{Rowe}, L. and {Houle}, D.",
 "maynardsmith1973": "{Maynard Smith}, J. and {Price}, G. R.",
 "vonrueden2008": "{von Rueden}, C. and {Gurven}, M. and {Kaplan}, H.",
 "barban2016": "{Barban}, N. and {Jansen}, R. and {de Vlaming}, R. and others",
 "maynardsmith1995": "{Maynard Smith}, J. and {Harper}, D. G. C.",
 "hanley1983": "{Hanley}, J. A. and {Lippman-Hand}, A.",
}
FIX = {
 "garciaruiz2016": {"pages": "E3995--E4004"}, "berna2012": {"pages": "E1215--E1220"}, "kong2017": {"pages": "E727--E732"},
 "henshaw2016": {"pages": "E300--E308"}, "milgrom1981": {"pages": "380--391"}, "shao2021": {"pages": "2535--2544"},
 "decarli2024": {"pages": "471--480"}, "lidborg2022pre": {"journal": "PsyArXiv (preprint)"},
 "byars2010": {"number": "suppl.~1"},
 "woodburn1982": {"pages": "431--451"}, "sweeney2002": {"pages": "132--147"},
 "hanley1983": {"journal": "{JAMA}", "pages": "1743--1745", "title": "{If nothing goes wrong, is everything all right? Interpreting zero numerators}"},
 "miller2000ch": {"booktitle": "The Nature of Intelligence (Novartis Foundation Symposium 233)", "publisher": "Wiley"},
 "boehm1993": {"title": "{Egalitarian behavior and reverse dominance hierarchy}"},
 "beals1984": {"title": "{Brain size, cranial morphology, climate, and time machines}"},
 "vonrueden2008": {"title": "{The multiple dimensions of male social status in an Amazonian society}"},
 "gintis2015": {"title": "{\\textit{Zoon Politikon}: The evolutionary origins of human political systems}"},
 "garfield2019": {"title": "{Evolutionary models of leadership: Tests and synthesis}"},
 "gurvenhill2009": {"title": "{Why do men hunt? A reevaluation of ``Man the Hunter'' and the sexual division of labor}"},
 "lewlevy2017": {"title": "{How do hunter-gatherer children learn subsistence skills? A meta-ethnographic review}"},
 "marlowe1998": {"title": "{The nubility hypothesis: The human breast as an honest signal of residual reproductive value}"},
 "berger2015": {"title": "{\\textit{Homo naledi}, a new species of the genus \\textit{Homo} from the Dinaledi Chamber, South Africa}"},
 "dirks2017": {"title": "{The age of \\textit{Homo naledi} and associated sediments in the Rising Star Cave, South Africa}"},
 "hirshleifer1989": {"journal": "Public Choice"},
 "berna2012": {"pages": "E1215--E1220", "title": "{Microstratigraphic evidence of in situ fire in the Acheulean strata of Wonderwerk Cave, Northern Cape province, South Africa}"},
 "sala2015": {"title": "{Lethal interpersonal violence in the Middle Pleistocene}"},
 "jaubert2016": {"title": "{Early Neanderthal constructions deep in Bruniquel Cave in southwestern France}"},
 "powell2009": {"title": "{Late Pleistocene demography and the appearance of modern human behavior}"},
 "decarli2024": {"pages": "471--480", "title": "{Trends in intracranial and cerebral volumes of Framingham Heart Study participants born 1930 to 1970}"},
 "crist2025": {"title": "{Herding with the hounds: The game of fifty-eight holes in the Ab\c{s}eron Peninsula}"},
 "rollefson1992": {"title": "{A Neolithic game board from `Ain Ghazal, Jordan}"},
 "deboer2001": {"title": "{Of dice and women: Gambling and exchange in Native North America}"},
 "hassett2018": {"title": "{Radical `royals'? Burial practices at Ba\c{s}ur H\"oy\"uk and the emergence of early states in Mesopotamia}"},
 "borkenau2004": {"title": "{Thin slices of behavior as cues of personality and intelligence}"},
 "judge2004": {"title": "{Intelligence and leadership: A quantitative review and test of theoretical propositions}"},
 "lahr2016": {"title": "{Inter-group violence among early Holocene hunter-gatherers of West Turkana, Kenya}"},
 "zohar2022": {"title": "{Evidence for the cooking of fish 780,000 years ago at Gesher Benot Ya'aqov, Israel}"},
 "kirkpatrick1990": {"title": "{The Darwin--Fisher theory of sexual selection in monogamous birds}"},
 "wiessner2014": {"title": "{Embers of society: Firelight talk among the Ju/'hoansi Bushmen}"},
}
MANUAL = r"""
@book{miller2000book, author={{Miller}, G. F.}, title={{The Mating Mind: How Sexual Choice Shaped the Evolution of Human Nature}}, publisher={Doubleday}, address={New York}, year={2000}}
@book{lush1937, author={{Lush}, J. L.}, title={{Animal Breeding Plans}}, publisher={Iowa State College Press}, address={Ames, IA}, year={1937}}
@book{falconer1996, author={{Falconer}, D. S. and {Mackay}, T. F. C.}, title={{Introduction to Quantitative Genetics}}, edition={4th}, publisher={Longman}, address={Harlow}, year={1996}}
@book{bulmer1980, author={{Bulmer}, M. G.}, title={{The Mathematical Theory of Quantitative Genetics}}, publisher={Clarendon Press}, address={Oxford}, year={1980}}
@incollection{alexander1989, author={{Alexander}, R. D.}, title={{Evolution of the human psyche}}, booktitle={The Human Revolution}, editor={P. Mellars and C. Stringer}, pages={455--513}, publisher={Edinburgh University Press}, year={1989}}
@article{crow1958, author={{Crow}, J. F.}, title={{Some possibilities for measuring selection intensities in man}}, journal={Hum. Biol.}, volume={30}, number={1}, pages={1--13}, year={1958}}
@article{chick2012, author={{Chick}, G. and {Yarnal}, C. and {Purrington}, A.}, title={{Play and mate preference: Testing the signal theory of adult playfulness}}, journal={Am. J. Play}, volume={4}, number={4}, pages={407--440}, year={2012}}
@book{boehm1999, author={{Boehm}, C.}, title={{Hierarchy in the Forest: The Evolution of Egalitarian Behavior}}, publisher={Harvard University Press}, address={Cambridge, MA}, year={1999}}
@book{crist2016, author={{Crist}, W. and {Dunn-Vaturi}, A.-E. and {de Voogt}, A.}, title={{Ancient Egyptians at Play: Board Games Across Borders}}, publisher={Bloomsbury}, address={London}, year={2016}}
@book{decker1992, author={{Decker}, W.}, title={{Sports and Games of Ancient Egypt}}, publisher={Yale University Press}, address={New Haven}, year={1992}}
@book{poliakoff1987, author={{Poliakoff}, M. B.}, title={{Combat Sports in the Ancient World: Competition, Violence, and Culture}}, publisher={Yale University Press}, address={New Haven}, year={1987}}
@misc{gss2024, author={{Davern}, M. and {Bautista}, R. and {Freese}, J. and {Herd}, P. and {Morgan}, S. L.}, title={{General Social Survey 1972--2024}}, howpublished={Machine-readable cumulative data file. NORC at the University of Chicago, Chicago. \url{https://gss.norc.org}}, year={2025}}
@misc{taylor2026iina, author={{Taylor}, J.}, title={{Intelligence is not authority: A bound on AI catastrophe with no term for what the machine wants, and six measurable conditions for making it small}}, howpublished={Companion manuscript (preprint, not peer reviewed)}, year={2026}}
@misc{healygssr, author={{Healy}, K.}, title={{gssr: General Social Survey data for use in R}}, howpublished={R package, version 0.9. \url{https://kjhealy.github.io/gssr/}}, year={2025}}
@incollection{finkel2007, author={{Finkel}, I. L.}, title={{On the rules for the Royal Game of Ur}}, booktitle={Ancient Board Games in Perspective}, editor={I. L. Finkel}, pages={16--32}, publisher={British Museum Press}, address={London}, year={2007}}

@book{baumeister2010,
  author = {Baumeister, R. F.},
  title = {Is There Anything Good About Men? {H}ow Cultures Flourish by Exploiting Men},
  publisher = {Oxford University Press},
  address = {New York},
  year = {2010}
}

@book{spencer1988,
  author = {Spencer, P.},
  title = {The {M}aasai of {M}atapato: A Study of Rituals of Rebellion},
  publisher = {Manchester University Press},
  address = {Manchester},
  year = {1988}
}

@book{frayser1985,
  author = {Frayser, S. G.},
  title = {Varieties of Sexual Experience: An Anthropological Perspective on Human Sexuality},
  publisher = {HRAF Press},
  address = {New Haven},
  year = {1985}
}

@book{binford2001,
  author = {Binford, L. R.},
  title = {Constructing Frames of Reference: An Analytical Method for Archaeological Theory Building Using Ethnographic and Environmental Data Sets},
  publisher = {University of California Press},
  address = {Berkeley},
  year = {2001}
}

@book{sanday1981book,
  author = {Sanday, P. R.},
  title = {Female Power and Male Dominance: On the Origins of Sexual Inequality},
  publisher = {Cambridge University Press},
  address = {Cambridge},
  year = {1981}
}

@article{white1988,
  author = {White, D. R.},
  title = {Rethinking polygyny: co-wives, codes, and cultural systems},
  journal = {Current Anthropology},
  volume = {29},
  pages = {529--572},
  year = {1988}
}

@misc{maasaiolympics,
  author = {{Big Life Foundation}},
  title = {Maasai {O}lympics},
  howpublished = {\url{https://biglife.org/maasai-olympics/maasai-olympics}},
  year = {2026},
  note = {Organizer's description, read 10 November 2026}
}

@misc{aljazeera2022,
  author = {{Al Jazeera}},
  title = {At {K}enya's {M}aasai {O}lympics, warriors swap lion hunt for high jump},
  howpublished = {\url{https://www.aljazeera.com/news/2022/12/11/at-kenyas-maasai-olympics-warriors-swap-lion-hunts-for-sports}},
  year = {2022},
  note = {News report, 11 December 2022}
}

@misc{aljazeera2023,
  author = {{Al Jazeera}},
  title = {Kenya's young {M}aasai reconnect with their culture at {E}unoto ceremony},
  howpublished = {\url{https://www.aljazeera.com/gallery/2023/8/21/kenyas-young-maasai-reconnect-with-their-culture-at-eunoto-ceremony}},
  year = {2023},
  note = {News report, 21 August 2023}
}

@misc{hinton2015,
  author = {Hinton, G. and Vinyals, O. and Dean, J.},
  title = {Distilling the knowledge in a neural network},
  howpublished = {arXiv:1503.02531},
  year = {2015}
}
"""

def write_bib():
    recs = {}
    for key, doi in DOI.items():
        recs[key] = by_doi(key, doi)
    for key, (t, fam, yr) in Q.items():
        recs[key] = by_query(key, t, fam, yr)
    out, changed = [], []
    for key, m in recs.items():
        if not m:
            print("MISSING", key); continue
        fix = dict(FIX.get(key, {}))
        if key in AUTH:
            fix["author"] = AUTH[key]
        e = entry(key, m, fix)
        if "title" not in FIX.get(key, {}):
            mt = re.search(r"  title = \{\{(.*)\}\}", e)
            if mt and needs_case(mt.group(1)):
                new = sentence(mt.group(1)); changed.append((key, new))
                e = e.replace(mt.group(0), "  title = {{%s}}" % new)
        e = e.replace("{Mulder}, M. B.", "{Borgerhoff Mulder}, M.").replace("☆", "")
        e = re.sub(r"\{([A-Z][A-ZÀ-Ü\-' ]{3,})\},", lambda g: "{" + g.group(1).title() + "},", e)
        out.append(e)
    open(os.path.join(ROOT, "paper", "refs.bib"), "w").write("% Generated by code/build_bib.py from Crossref records; do not edit by hand.\n" + "\n".join(out) + MANUAL)
    print("wrote", len(out), "entries +", MANUAL.count("@"), "manual")
    for k, t in changed:
        print("  CASED", k, "::", t[:95])

if "--bib" in sys.argv:
    write_bib()
