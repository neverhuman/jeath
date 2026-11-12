"""Render the performance-before-outcomes study figures, tables, macros and sequential bibliography."""
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

ROOT = Path(__file__).resolve().parents[1]
V9_ROOT = ROOT / "project/brain/tips/Performance_Before_Outcomes_v9"
FACE_CMAP = LinearSegmentedColormap.from_list(
    "pbo_face_div",
    [
        (0.00, "#08306B"),
        (0.22, "#1B5E8A"),
        (0.42, "#7FB3D5"),
        (0.50, "#F4F5F7"),
        (0.58, "#F5B041"),
        (0.78, "#B84A27"),
        (1.00, "#7F1D1D"),
    ],
    N=256,
)

# Unified semantic editorial palette (Law V of docs/PAPER_MISSION.md)
INK = "#18222F"
INK2 = "#4A5565"
MUTED = "#7A7872"
GRID = "#E5E4DF"
BAND = "#F3F1EC"
WOMEN = "#B84A27"   # Terracotta-rust (diamonds)
MEN = "#1B5E8A"     # Deep cerulean-navy (circles)
AQUA = "#0F6E66"    # Competence / capacity / energetic release
VIOLET = "#5C4B99"  # Delivery / commitment / contested genomic
GOLD = "#C87D16"    # Static cue / framing / unlinked physical
RED = "#C0392B"
B300 = "#6A9EC9"
B550 = "#1B5E8A"
GREY = "#7A7872"

SERIES = (("Female", "Women", WOMEN, "D"), ("Male", "Men", MEN, "o"))
FONT_FAMILY = "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 7.4,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlesize": 8.2,
    "axes.titleweight": "bold",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 6.8,
    "lines.solid_capstyle": "round",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
})


EXTRA_BIBITEMS = r"""
\bibitem[{Whyte} et~al.(2021){Whyte}, {Brooks}, {Chan}, and {Torgler}]{whyte2021}
S.~{Whyte}, R.~C. {Brooks}, H.~F. {Chan}, and B.~{Torgler}.
\newblock {Sex differences in sexual attraction for aesthetics, resources and personality across age}.
\newblock \emph{PLOS ONE}, 16\penalty0 (5):\penalty0 e0250151, 2021.
\newblock \doi{10.1371/journal.pone.0250151}.

\bibitem[{Havl{\'\i}{\v c}ek} et~al.(2020){Havl{\'\i}{\v c}ek}, {Winternitz}, and {Roberts}]{havlicek2020}
J.~{Havl{\'\i}{\v c}ek}, J.~{Winternitz}, and S.~C. {Roberts}.
\newblock {Major histocompatibility complex-associated odour preferences and human mate choice: near and far horizons}.
\newblock \emph{Philosophical Transactions of the Royal Society B}, 375\penalty0 (1800):\penalty0 20190260, 2020.
\newblock \doi{10.1098/rstb.2019.0260}.

\bibitem[{Winternitz} et~al.(2017){Winternitz}, {Abbate}, {Huchard}, {Havl{\'\i}{\v c}ek}, and {Garamszegi}]{winternitz2017}
J.~{Winternitz}, J.~L. {Abbate}, E.~{Huchard}, J.~{Havl{\'\i}{\v c}ek}, and L.~Z. {Garamszegi}.
\newblock {Patterns of MHC-dependent mate selection in humans and nonhuman primates: a meta-analysis}.
\newblock \emph{Molecular Ecology}, 26\penalty0 (2):\penalty0 668--688, 2017.
\newblock \doi{10.1111/mec.13920}.

\bibitem[{Kunkle} et~al.(2025){Kunkle}, {Tennyson}, {Wander}, {Shell-Duncan}, and {Eisenberg}]{kunkle2025}
A.~E. {Kunkle}, R.~L. {Tennyson}, K.~{Wander}, B.~{Shell-Duncan}, and D.~T.~A. {Eisenberg}.
\newblock {The attention-deficit/hyperactivity disorder-associated DRD4 7R allele predicts household economic status but not nutritional status in Northern Kenyan Rendille children}.
\newblock \emph{American Journal of Human Biology}, 37\penalty0 (3):\penalty0 e70027, 2025.
\newblock \doi{10.1002/ajhb.70027}.

\bibitem[{Stulp} et~al.(2012){Stulp}, {Pollet}, {Verhulst}, and {Buunk}]{stulp2012}
G.~{Stulp}, T.~V. {Pollet}, S.~{Verhulst}, and A.~P. {Buunk}.
\newblock {A curvilinear effect of height on reproductive success in human males}.
\newblock \emph{Behavioral Ecology and Sociobiology}, 66\penalty0 (3):\penalty0 375--384, 2012.
\newblock \doi{10.1007/s00265-011-1283-2}.

\bibitem[{Stulp} et~al.(2013){Stulp}, {Buunk}, {Kurzban}, and {Verhulst}]{stulp2013}
G.~{Stulp}, A.~P. {Buunk}, R.~{Kurzban}, and S.~{Verhulst}.
\newblock {The height of choosiness: mutual mate choice for stature results in suboptimal pair formation for both sexes}.
\newblock \emph{Animal Behaviour}, 86\penalty0 (1):\penalty0 37--46, 2013.
\newblock \doi{10.1016/j.anbehav.2013.03.038}.

\bibitem[{Pawlowski} et~al.(2000){Pawlowski}, {Dunbar}, and {Lipowicz}]{pawlowski2000}
B.~{Pawlowski}, R.~I.~M. {Dunbar}, and A.~{Lipowicz}.
\newblock {Tall men have more reproductive success}.
\newblock \emph{Nature}, 403\penalty0 (6766):\penalty0 156, 2000.
\newblock \doi{10.1038/35003107}.

\bibitem[{Sharma} et~al.(2024){Sharma}, {Tong}, {Korbak}, {Duvenaud}, {Askell}, {Bowman}, {Cheng}, {Durmus}, {Hatfield-Dodds}, {Johnston}, {Kravec}, {Maxwell}, {McCandlish}, {Ndousse}, {Rausch}, {Schiefer}, {Yan}, {Zhang}, and {Perez}]{sharma2024}
M.~{Sharma}, M.~{Tong}, T.~{Korbak}, et~al.
\newblock {Towards understanding sycophancy in language models}.
\newblock In \emph{International Conference on Learning Representations (ICLR)}, 2024.

\bibitem[{Gao} et~al.(2023){Gao}, {Schulman}, and {Hilton}]{gao2023}
L.~{Gao}, J.~{Schulman}, and J.~{Hilton}.
\newblock {Scaling laws for reward model overoptimization}.
\newblock In \emph{Proceedings of Machine Learning Research}, volume 202, pages 10835--10866, 2023.

\bibitem[{Langlois} and {Roggman}(1990)]{langlois1990}
J.~H. {Langlois} and L.~A. {Roggman}.
\newblock {Attractive faces are only average}.
\newblock \emph{Psychological Science}, 1\penalty0 (2):\penalty0 115--121, 1990.
\newblock \doi{10.1111/j.1467-9280.1990.tb00079.x}.

\bibitem[{Little} et~al.(2011){Little}, {Jones}, and {DeBruine}]{little2011}
A.~C. {Little}, B.~C. {Jones}, and L.~M. {DeBruine}.
\newblock {Facial attractiveness: evolutionary based research}.
\newblock \emph{Philosophical Transactions of the Royal Society B}, 366\penalty0 (1571):\penalty0 1638--1659, 2011.
\newblock \doi{10.1098/rstb.2010.0404}.

\bibitem[{Deng} et~al.(2019){Deng}, {Guo}, {Xue}, and {Zafeiriou}]{deng2019arcface}
J.~{Deng}, J.~{Guo}, N.~{Xue}, and S.~{Zafeiriou}.
\newblock {ArcFace: Additive angular margin loss for deep face recognition}.
\newblock In \emph{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, pages 4690--4699, 2019.
\newblock \doi{10.1109/CVPR.2019.00482}.

\bibitem[{Woo} et~al.(2023){Woo}, {Debnath}, {Hu}, {Chen}, {Liu}, {Kweon}, and {Xie}]{woo2023convnextv2}
S.~{Woo}, S.~{Debnath}, R.~{Hu}, X.~{Chen}, Z.~{Liu}, I.~S. {Kweon}, and S.~{Xie}.
\newblock {ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders}.
\newblock In \emph{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, pages 16133--16142, 2023.
\newblock \doi{10.1109/CVPR52729.2023.01548}.

\bibitem[{Oquab} et~al.(2024){Oquab}, {Darcet}, {Moutakanni}, et~al.]{oquab2024dinov2}
M.~{Oquab}, T.~{Darcet}, T.~{Moutakanni}, et~al.
\newblock {DINOv2: Learning robust visual features without supervision}.
\newblock \emph{Transactions on Machine Learning Research}, 2024.

\bibitem[{Lundberg} and {Lee}(2017)]{lundberg2017shap}
S.~M. {Lundberg} and S.-I. {Lee}.
\newblock {A unified approach to interpreting model predictions}.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, volume~30, pages 4765--4774, 2017.

\bibitem[{Lisiecki} and {Raymo}(2005)]{lisiecki2005}
L.~E. {Lisiecki} and M.~E. {Raymo}.
\newblock {A Pliocene--Pleistocene stack of 57 globally distributed benthic $\delta^{18}$O records}.
\newblock \emph{Paleoceanography}, 20\penalty0 (1):\penalty0 PA1003, 2005.
\newblock \doi{10.1029/2004PA001071}.

\bibitem[{Aiello} and {Wheeler}(1995)]{aiello1995}
L.~C. {Aiello} and P.~{Wheeler}.
\newblock {The expensive-tissue hypothesis: the brain and the digestive system in human and primate evolution}.
\newblock \emph{Current Anthropology}, 36\penalty0 (2):\penalty0 199--221, 1995.
\newblock \doi{10.1086/204350}.

\bibitem[{Wrangham}(2009)]{wrangham2009}
R.~{Wrangham}.
\newblock \emph{{Catching Fire: How Cooking Made Us Human}}.
\newblock Basic Books, New York, 2009.

\bibitem[{Hublin} et~al.(2017){Hublin}, {Ben-Ncer}, {Bailey}, et~al.]{hublin2017}
J.-J. {Hublin}, A.~{Ben-Ncer}, S.~E. {Bailey}, et~al.
\newblock {New fossils from Jebel Irhoud, Morocco and the pan-African origin of \emph{Homo sapiens}}.
\newblock \emph{Nature}, 546\penalty0 (7657):\penalty0 289--292, 2017.
\newblock \doi{10.1038/nature22336}.

\bibitem[{Clark} et~al.(2006){Clark}, {Archer}, {Pollard}, et~al.]{clark2006}
P.~U. {Clark}, D.~{Archer}, D.~{Pollard}, et~al.
\newblock {The middle Pleistocene transition: characteristics, mechanisms, and implications for long-term changes in atmospheric pCO$_2$}.
\newblock \emph{Quaternary Science Reviews}, 25\penalty0 (23--24):\penalty0 3150--3184, 2006.
\newblock \doi{10.1016/j.quascirev.2006.07.008}.

\bibitem[{Chalk} et~al.(2017){Chalk}, {Hain}, {Foster}, et~al.]{chalk2017}
T.~B. {Chalk}, M.~P. {Hain}, G.~L. {Foster}, et~al.
\newblock {Causes of ice age intensification across the Mid-Pleistocene Transition}.
\newblock \emph{Proceedings of the National Academy of Sciences}, 114\penalty0 (50):\penalty0 13114--13119, 2017.
\newblock \doi{10.1073/pnas.1702143114}.

\bibitem[{Yan} et~al.(2019){Yan}, {Bender}, {Brook}, et~al.]{yan2019}
Y.~{Yan}, M.~L. {Bender}, E.~J. {Brook}, et~al.
\newblock {Two-million-year-old snapshots of atmospheric gases from Antarctic ice}.
\newblock \emph{Nature}, 574\penalty0 (7780):\penalty0 663--666, 2019.
\newblock \doi{10.1038/s41586-019-1692-3}.

\bibitem[{Pena} and {Goldstein}(2014)]{pena2014}
L.~D. {Pena} and S.~L. {Goldstein}.
\newblock {Thermohaline circulation crisis and impacts during the mid-Pleistocene transition}.
\newblock \emph{Science}, 345\penalty0 (6194):\penalty0 318--322, 2014.
\newblock \doi{10.1126/science.1249770}.

\bibitem[{Margari} et~al.(2023){Margari}, {Hodell}, {Parfitt}, et~al.]{margari2023}
V.~{Margari}, D.~A. {Hodell}, S.~A. {Parfitt}, et~al.
\newblock {Extreme glacial cooling likely led to hominin depopulation of Europe in the Early Pleistocene}.
\newblock \emph{Science}, 381\penalty0 (6658):\penalty0 693--699, 2023.
\newblock \doi{10.1126/science.adf4445}.

\bibitem[{Muttoni} and {Kent}(2024)]{muttoni2024}
G.~{Muttoni} and D.~V. {Kent}.
\newblock {Hominin population bottleneck coincided with migration from Africa during the Early Pleistocene ice age transition}.
\newblock \emph{Proceedings of the National Academy of Sciences}, 121\penalty0 (13):\penalty0 e2318903121, 2024.
\newblock \doi{10.1073/pnas.2318903121}.

\bibitem[{Timmermann} et~al.(2022){Timmermann}, {Yun}, {Raia}, et~al.]{timmermann2022}
A.~{Timmermann}, K.-S. {Yun}, P.~{Raia}, et~al.
\newblock {Climate effects on archaic human habitats and species successions}.
\newblock \emph{Nature}, 604\penalty0 (7906):\penalty0 495--501, 2022.
\newblock \doi{10.1038/s41586-022-04600-9}.

\bibitem[{Ao} et~al.(2026){Ao}, {Ruan}, {Mart{\'\i}nez-Garcia}, et~al.]{ao2026}
H.~{Ao}, J.~{Ruan}, A.~{Mart{\'\i}nez-Garcia}, et~al.
\newblock {Orbital-scale climate variability and Middle Pleistocene hominin evolution in East Asia}.
\newblock \emph{Quaternary Science Reviews}, 349:\penalty0 109128, 2026.

\bibitem[{Hu} et~al.(2023){Hu}, {Hao}, {Du}, et~al.]{hu2023}
W.~{Hu}, Z.~{Hao}, P.~{Du}, et~al.
\newblock {Genomic inference of a severe human bottleneck during the Early to Middle Pleistocene transition}.
\newblock \emph{Science}, 381\penalty0 (6661):\penalty0 979--984, 2023.
\newblock \doi{10.1126/science.abq7487}.

\bibitem[{Zhou} et~al.(2026){Zhou}, {Hu}, {Hao}, and {Li}]{zhou2026}
Y.~{Zhou}, W.~{Hu}, Z.~{Hao}, and H.~{Li}.
\newblock {Evaluating ancient human population bottlenecks and Mid-Pleistocene evolutionary transitions}.
\newblock \emph{Molecular Biology and Evolution}, 43\penalty0 (7):\penalty0 msag112, 2026.

\bibitem[{Deng} et~al.(2025){Deng}, {Nielsen}, and {Song}]{deng2025}
Y.~{Deng}, R.~{Nielsen}, and Y.~S. {Song}.
\newblock {Aapus and the statistical identifiability of severe ancient population bottlenecks from site frequency spectra}.
\newblock \emph{Genetics}, 229\penalty0 (1):\penalty0 iyae192, 2025.
\newblock \doi{10.1093/genetics/iyae192}.

\bibitem[{Cousins} and {Durvasula}(2025)]{cousins2025}
T.~{Cousins} and A.~{Durvasula}.
\newblock {Insufficient evidence for a severe bottleneck in humans during the Early to Middle Pleistocene transition}.
\newblock \emph{Molecular Biology and Evolution}, 42\penalty0 (2):\penalty0 msaf041, 2025.
\newblock \doi{10.1093/molbev/msaf041}.

\bibitem[{Terhorst}(2025)]{terhorst2025}
J.~{Terhorst}.
\newblock {Accelerated Bayesian inference of population size history from recombining sequence data}.
\newblock \emph{Nature Genetics}, 57:\penalty0 412--420, 2025.

\bibitem[{Ragsdale} et~al.(2023){Ragsdale}, {Weaver}, {Atkinson}, et~al.]{ragsdale2023}
A.~P. {Ragsdale}, T.~D. {Weaver}, E.~G. {Atkinson}, et~al.
\newblock {A weakly structured stem for human origins in Africa}.
\newblock \emph{Nature}, 617\penalty0 (7962):\penalty0 755--763, 2023.
\newblock \doi{10.1038/s41586-023-06055-y}.

\bibitem[{Cousins} et~al.(2024){Cousins}, {Scally}, and {Durbin}]{cousins2024}
T.~{Cousins}, A.~{Scally}, and R.~{Durbin}.
\newblock {A structured coalescent model reveals deep ancestral structure shared by all modern humans}.
\newblock \emph{Nature Genetics}, 57:\penalty0 856--864, 2024.
\newblock \doi{10.1038/s41588-025-02117-1}.

\bibitem[{Goren-Inbar} et~al.(2000){Goren-Inbar}, {Feibel}, {Verosub}, et~al.]{goreninbar2000}
N.~{Goren-Inbar}, C.~S. {Feibel}, K.~L. {Verosub}, et~al.
\newblock {Pleistocene milestones on the Out-of-Africa corridor at Gesher Benot Ya'aqov, Israel}.
\newblock \emph{Science}, 289\penalty0 (5481):\penalty0 944--947, 2000.
\newblock \doi{10.1126/science.289.5481.944}.

\bibitem[{Par{\'e}s} et~al.(2013){Par{\'e}s}, {Arnold}, {Duval}, et~al.]{pares2013}
J.~M. {Par{\'e}s}, L.~{Arnold}, M.~{Duval}, et~al.
\newblock {Reassessing the age of Atapuerca-TD6 (Spain): new paleomagnetic results}.
\newblock \emph{Journal of Archaeological Science}, 40\penalty0 (12):\penalty0 4586--4595, 2013.
\newblock \doi{10.1016/j.jas.2013.06.013}.

\bibitem[{Koeberl} et~al.(2007){Koeberl}, {Milkereit}, {Overpeck}, et~al.]{koeberl2007}
C.~{Koeberl}, B.~{Milkereit}, J.~T. {Overpeck}, et~al.
\newblock {An international and multidisciplinary drilling project into a young complex impact structure: the 2004 ICDP Bosumtwi impact crater project}.
\newblock \emph{Meteoritics \& Planetary Science}, 42\penalty0 (4--5):\penalty0 483--511, 2007.
\newblock \doi{10.1111/j.1945-5100.2007.tb01057.x}.

\bibitem[{Rochette} et~al.(2019){Rochette}, {Ala{\c c}}, {Beck}, et~al.]{rochette2019}
P.~{Rochette}, R.~{Ala{\c c}}, P.~{Beck}, et~al.
\newblock {Pantasma: evidence for a Pleistocene circa 14 km diameter impact crater in Nicaragua}.
\newblock \emph{Meteoritics \& Planetary Science}, 54\penalty0 (4):\penalty0 880--901, 2019.
\newblock \doi{10.1111/maps.13244}.

\bibitem[{Ito}(2020)]{ito2020}
H.~{Ito}.
\newblock {Uranium--lead dating of the Oldest Toba Tuff and Quaternary super-eruptions in Sumatra}.
\newblock \emph{Quaternary Geochronology}, 59:\penalty0 101095, 2020.

\bibitem[{Jourdan} et~al.(2019){Jourdan}, {Nomade}, {Wingate}, et~al.]{jourdan2019}
F.~{Jourdan}, S.~{Nomade}, M.~T.~D. {Wingate}, et~al.
\newblock {Ultraprecise age and formation temperature of splash-form Australasian tektites}.
\newblock \emph{Meteoritics \& Planetary Science}, 54\penalty0 (10):\penalty0 2573--2591, 2019.
\newblock \doi{10.1111/maps.13305}.

\bibitem[{Sieh} et~al.(2020){Sieh}, {Herrin}, {Jicha}, et~al.]{sieh2020}
K.~{Sieh}, J.~{Herrin}, B.~{Jicha}, et~al.
\newblock {Australasian impact crater buried under the Bolaven volcanic field, Southern Laos}.
\newblock \emph{Proceedings of the National Academy of Sciences}, 117\penalty0 (3):\penalty0 1346--1353, 2020.
\newblock \doi{10.1073/pnas.1904368116}.

\bibitem[{Sieh} et~al.(2024){Sieh}, {Herrin}, {Jicha}, et~al.]{sieh2024}
K.~{Sieh}, J.~{Herrin}, B.~{Jicha}, et~al.
\newblock {Proximal ejecta of the Bolaven extraterrestrial impact, southern Laos}.
\newblock \emph{Proceedings of the National Academy of Sciences}, 121\penalty0 (1):\penalty0 e2310351120, 2024.
\newblock \doi{10.1073/pnas.2310351120}.

\bibitem[{Mizera} and {Strunga}(2024)]{mizera2024}
J.~{Mizera} and V.~{Strunga}.
\newblock {Geochemical and geophysical constraints on the proposed Bolaven Plateau source crater for Australasian tektites}.
\newblock \emph{Earth-Science Reviews}, 252:\penalty0 104742, 2024.

\bibitem[{Michel} et~al.(2024){Michel}, {Shen}, {Moncel}, et~al.]{michel2024}
V.~{Michel}, G.~{Shen}, M.-H. {Moncel}, et~al.
\newblock {Early Middle Pleistocene bifacial industries and tektite-bearing horizons in the Bose Basin and mainland Southeast Asia}.
\newblock \emph{Quaternary Science Reviews}, 328:\penalty0 108532, 2024.

\bibitem[{Singer} et~al.(2019){Singer}, {Jicha}, {Mochizuki}, et~al.]{singer2019}
B.~S. {Singer}, B.~R. {Jicha}, N.~{Mochizuki}, et~al.
\newblock {Synchronizing volcanic, sedimentary, and ice core records of Earth's last magnetic polarity reversal}.
\newblock \emph{Science Advances}, 5\penalty0 (8):\penalty0 eaaw4621, 2019.
\newblock \doi{10.1126/sciadv.aaw4621}.

\bibitem[{Blurton Jones} et~al.(2000){Blurton Jones}, {Marlowe}, {Hawkes}, and {O'Connell}]{blurtonjones2000}
N.~G. {Blurton Jones}, F.~W. {Marlowe}, K.~{Hawkes}, and J.~F. {O'Connell}.
\newblock {Paternal investment and hunter-gatherer divorce rates}.
\newblock In L.~{Cronk}, N.~{Chagnon}, and W.~{Irons}, editors, \emph{Adaptation and Human Behavior: An Anthropological Perspective}, pages 69--90. Aldine de Gruyter, New York, 2000.

\bibitem[{Scelza}(2013)]{scelza2013}
B.~A. {Scelza}.
\newblock {Choosy but not chaste: multiple mating in human females}.
\newblock \emph{Evolutionary Anthropology}, 22\penalty0 (5):\penalty0 259--269, 2013.
\newblock \doi{10.1002/evan.21373}.

\bibitem[{Collier} and {Rosaldo}(1981)]{collier1981}
J.~F. {Collier} and M.~Z. {Rosaldo}.
\newblock {Politics and gender in simple societies}.
\newblock In S.~B. {Ortner} and H.~{Whitehead}, editors, \emph{Sexual Meanings: The Cultural Construction of Gender and Sexuality}, pages 275--329. Cambridge University Press, Cambridge, 1981.

\bibitem[{Joel} et~al.(2020){Joel}, {Eastwick}, {Allison}, {Arriaga}, {Baker}, {Bar-Kalifa}, {Bergeron}, {Birnbaum}, {Brock}, {Brumbaugh}, et~al.]{joel2020}
S.~{Joel}, P.~W. {Eastwick}, C.~J. {Allison}, et~al.
\newblock {Machine learning uncovers the most robust self-report predictors of relationship quality across 43 longitudinal couples studies}.
\newblock \emph{Proceedings of the National Academy of Sciences}, 117\penalty0 (32):\penalty0 19061--19071, 2020.
\newblock \doi{10.1073/pnas.1917036117}.

\bibitem[{Eastwick} and {Hunt}(2014)]{eastwick2014}
P.~W. {Eastwick} and L.~L. {Hunt}.
\newblock {Relational mate value: consensus and uniqueness in romantic evaluations}.
\newblock \emph{Journal of Personality and Social Psychology}, 106\penalty0 (5):\penalty0 728--751, 2014.
\newblock \doi{10.1037/a0035884}.

\bibitem[{Sternberg}(1986)]{sternberg1986}
R.~J. {Sternberg}.
\newblock {A triangular theory of love}.
\newblock \emph{Psychological Review}, 93\penalty0 (2):\penalty0 119--135, 1986.
\newblock \doi{10.1037/0033-295X.93.2.119}.

\bibitem[{Frank}(1988)]{frank1988}
R.~H. {Frank}.
\newblock \emph{{Passions Within Reason: The Strategic Role of the Emotions}}.
\newblock W.~W. Norton, New York, 1988.

\bibitem[{Rusbult} et~al.(1998){Rusbult}, {Martz}, and {Agnew}]{rusbult1998}
C.~E. {Rusbult}, J.~M. {Martz}, and C.~R. {Agnew}.
\newblock {The Investment Model Scale: measuring commitment level, satisfaction level, quality of alternatives, and investment size}.
\newblock \emph{Personal Relationships}, 5\penalty0 (4):\penalty0 357--387, 1998.
\newblock \doi{10.1111/j.1475-6811.1998.tb00177.x}.

\bibitem[{Betzig}(1989)]{betzig1989}
L.~{Betzig}.
\newblock {Causes of conjugal dissolution: a cross-cultural study}.
\newblock \emph{Current Anthropology}, 30\penalty0 (5):\penalty0 654--676, 1989.
\newblock \doi{10.1086/203798}.

\bibitem[{Jankowiak} and {Fischer}(1992)]{jankowiak1992}
W.~R. {Jankowiak} and E.~F. {Fischer}.
\newblock {A cross-cultural perspective on romantic love}.
\newblock \emph{Ethnology}, 31\penalty0 (2):\penalty0 149--155, 1992.
\newblock \doi{10.2307/3773618}.

\bibitem[{Isler} and {van Schaik}(2012)]{isler2012}
K.~{Isler} and C.~P. {van Schaik}.
\newblock {Allomaternal care, life history and brain size evolution in mammals}.
\newblock \emph{Current Anthropology}, 53\penalty0 (S6):\penalty0 S453--S465, 2012.
\newblock \doi{10.1086/667623}.

\bibitem[{Hrdy}(2009)]{hrdy2009}
S.~B. {Hrdy}.
\newblock \emph{{Mothers and Others: The Evolutionary Origins of Mutual Understanding}}.
\newblock Harvard University Press, Cambridge, MA, 2009.

\bibitem[{Mehr} et~al.(2019){Mehr}, {Singh}, {Knox}, {Ketter}, {Pickens-Jones}, {Atwood}, {Lucas}, {Jacoby}, {Egner}, {Hopkins}, et~al.]{mehr2019}
S.~A. {Mehr}, M.~{Singh}, D.~{Knox}, et~al.
\newblock {Universality and diversity in human song}.
\newblock \emph{Science}, 366\penalty0 (6468):\penalty0 eaax0868, 2019.
\newblock \doi{10.1126/science.aax0868}.

\bibitem[{Conroy-Beam} et~al.(2016){Conroy-Beam}, {Goetz}, and {Buss}]{conroybeam2016}
D.~{Conroy-Beam}, C.~D. {Goetz}, and D.~M. {Buss}.
\newblock {What predicts romantic relationship satisfaction and mate retention intensity? Mate preference fulfillment or mate value discrepancies?}
\newblock \emph{Evolution and Human Behavior}, 37\penalty0 (6):\penalty0 440--448, 2016.
\newblock \doi{10.1016/j.evolhumbehav.2016.04.003}.

\bibitem[{Finkel} et~al.(2012){Finkel}, {Eastwick}, {Karney}, {Reis}, and {Sprecher}]{finkel2012}
E.~J. {Finkel}, P.~W. {Eastwick}, B.~R. {Karney}, H.~T. {Reis}, and S.~{Sprecher}.
\newblock {Online dating: a critical analysis from the perspective of psychological science}.
\newblock \emph{Psychological Science in the Public Interest}, 13\penalty0 (1):\penalty0 3--66, 2012.
\newblock \doi{10.1177/1529100612436522}.

\bibitem[{Kim} and {Escobedo-Land}(2015)]{kim2015}
A.~Y. {Kim} and A.~{Escobedo-Land}.
\newblock {OkCupid profile data for introductory statistics and data science courses}.
\newblock \emph{Journal of Statistics Education}, 23\penalty0 (2):\penalty0 1--25, 2015.
\newblock \doi{10.1080/10691898.2015.11889737}.
"""


def pdf(fig, dpi=None):
    stream = BytesIO()
    save_kw = {
        "format": "pdf",
        "bbox_inches": "tight",
        "pad_inches": 0.025,
        "metadata": {"Creator": "scripts/paper.py", "CreationDate": None},
    }
    if dpi is not None:
        save_kw["dpi"] = dpi
    fig.savefig(stream, **save_kw)
    plt.close(fig)
    return stream.getvalue()


def panel(ax, letter, title, xlabel=None, ylabel=None, percent=False, pad=6, fontsize=8.1):
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", fontsize=fontsize, pad=pad)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=3.5)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=3.5)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.55)


def dot(ax, x, y, color, marker="o", s=16, z=3, alpha=1.0, **kw):
    return ax.scatter(x, y, s=s, marker=marker, facecolor=color, edgecolor="white", linewidth=0.45, zorder=z, alpha=alpha, **kw)


def age_axis(ax):
    ax.set_xlim(17, 80)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 79])
    ax.set_xlabel("Recorded age", labelpad=3.5)


def legend_top(fig, handles, y=1.015, ncol=None):
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, y),
        ncol=ncol or len(handles),
        columnspacing=1.4,
        handlelength=1.9,
        fontsize=7.3,
    )


def fossil_step_curve(fossils, x):
    st, bp = fossils["step"], fossils["bps"]
    l1, l3, le = math.log(st["level_I"]), math.log(st["level_III"]), math.log(st["end"])
    y = np.where(
        x >= bp[0],
        l1,
        np.where(
            x >= bp[1],
            l1 + (l3 - l1) * (bp[0] - x) / (bp[0] - bp[1]),
            np.where(x >= bp[2], l3, l3 + (le - l3) * (bp[2] - x) / (bp[2] - 0.01)),
        ),
    )
    return np.exp(y)


# ---------------------------------------------------------------------------
# 1. PAGE-1 FLAGSHIP 3-PANEL VISUAL SYNTHESIS (performance-fig1-main.pdf)
# ---------------------------------------------------------------------------
def fig1_main_plot(data):
    osyn = data["open_synthesis"]
    v9 = osyn["v9_anchors"]
    fossils = v9["fossils"]
    kowal = osyn["kowal_2025"]
    joel = osyn["joel_2020"]
    pp = data["paleoclimate_plateau"]
    pov = data["pov"]
    cc = osyn["childhood_coevolution"]
    hcm = osyn["hcmst_2017"]
    chk_hw = data["speed_dating"]["checklist_and_hindsight"]["hobby_checklist_vs_live_wavelength"]
    chk_kow = kowal["checklist_and_surplus_decomposition"]["pooled"]["predictors_sd_per_sd"]
    pvc_gap = 100.0 * data["holoface_benchmark"]["psychometric_voter_ceiling"]["tiers"]["votes_500_plus"]["irreducible_single_context_gap_full"]

    fig = plt.figure(figsize=(7.22, 2.94))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.15, 1.08], wspace=0.36)
    fig.subplots_adjust(left=0.068, right=0.992, bottom=0.155, top=0.880)

    # Panel A: Brain Growth, 800-kyr Stasis (~908 cm^3), & Cooking + Childhood Pair-Bond Release
    a_gs = outer[0, 0].subgridspec(2, 1, height_ratios=[1.92, 1.0], hspace=0.10)
    ax_a = fig.add_subplot(a_gs[0, 0])
    ax_lr = fig.add_subplot(a_gs[1, 0], sharex=ax_a)
    for ax_sub in (ax_a, ax_lr):
        ax_sub.axvspan(1.50, 0.70, color="#ECE8DD", zorder=0, lw=0)
        ax_sub.axvspan(0.93, 0.813, color=VIOLET, alpha=0.12, zorder=1, lw=0)
        ax_sub.axvline(1.12, color=MEN, lw=0.75, ls=":", alpha=0.85, zorder=2)
    gstyle = {
        "aus": (GREY, "o", "Australopiths"),
        "eh": (B300, "s", "Early Homo"),
        "mp": (B550, "D", "Mid-Pleist."),
        "nea": (WOMEN, "^", "Neanderthals"),
        "sap": (AQUA, "o", "H. sapiens"),
    }
    for k in ("aus", "eh", "mp", "sap", "nea"):
        pts = np.asarray(fossils["groups"][k], dtype=float)
        c, m, _ = gstyle[k]
        dot(ax_a, pts[:, 0], pts[:, 1], c, m, s=11 if m != "D" else 9, z=4 if k == "nea" else 3)
    xx = np.linspace(3.28, 0.01, 500)
    ax_a.plot(xx, fossil_step_curve(fossils, xx), color=INK, lw=1.7, zorder=5)
    ax_a.annotate(
        "", xy=(1.12, 1820), xytext=(1.50, 1820),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.75),
    )
    ax_a.text(1.31, 1980, "380\u2013711 kyr lead", fontsize=5.1, color=INK, ha="center", va="bottom", fontweight="bold")
    ax_a.text(
        1.12, 1120, "Stalled 800 kyr\n(~908 cm\u00b3)", fontsize=5.2, color=INK, ha="center", va="center", fontweight="bold",
        linespacing=1.02, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )
    ax_a.text(
        2.58, 1380,
        f"Age 4.5: {cc['kuzawa_2014_brain_metabolism']['peak_rmr_pct']:.0f}% RMR\nSCCS: 0% solo mom",
        fontsize=4.8, color=VIOLET, ha="center", va="center", fontweight="bold", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    for xf in (1.0, 0.78):
        ax_a.plot([xf], [318], marker="^", ms=4.0, color=AQUA, zorder=6)
    ax_a.text(0.82, 540, "Fire + love\n(+50%/Myr)", fontsize=4.9, color=AQUA, ha="center", fontweight="bold", linespacing=1.0)
    dot(ax_a, [0.08], [426], VIOLET, "o", s=14, z=5)
    ax_a.text(0.40, 340, "dwarfs", fontsize=4.9, color=INK2, ha="center")
    ax_a.set_yscale("log")
    ax_a.set_xlim(3.32, -0.16)
    ax_a.set_ylim(285, 2950)
    ax_a.set_yticks([400, 800, 1600])
    ax_a.set_yticklabels(["400", "800", "1,600"])
    ax_a.minorticks_off()
    plt.setp(ax_a.get_xticklabels(), visible=False)
    panel(ax_a, "A", "When brains grew vs. stalled", ylabel="cm$^3$ (log)", fontsize=7.6, pad=4)

    lr = pp["lr04_plot_series"]
    ax_lr.fill_between(lr["envelope_age_ma"], lr["envelope_p10"], lr["envelope_p90"], color=B300, alpha=0.22, lw=0)
    ax_lr.plot(lr["age_ma"], lr["d18o_permil"], color=MEN, lw=0.6, alpha=0.82)
    ax_lr.plot(lr["envelope_age_ma"], lr["envelope_mean"], color=INK, lw=1.05)
    ax_lr.set_ylim(2.68, 5.48)
    ax_lr.set_yticks([3.2, 4.6])
    ax_lr.set_xticks([3, 2, 1, 0])
    ax_lr.grid(axis="y", color=GRID, lw=0.5)
    ax_lr.set_xlabel("Million years ago (Ma)", labelpad=3.0)
    ax_lr.set_ylabel("$\\delta^{18}$O (\u2030)", fontsize=6.4, labelpad=2.5)
    ax_lr.text(1.16, 5.02, "MIS 34", fontsize=4.9, color=MEN, ha="right", fontweight="bold")
    ax_lr.text(0.74, 2.98, "Ne~1,280 (contested)", fontsize=4.6, color=VIOLET, ha="right")

    # Panel B: Secondary Entry Filters vs. Primary Pair-Bond Criteria (90.5k + 47.9k)
    b_gs = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.42)
    ax_b1 = fig.add_subplot(b_gs[0, 0])
    ax_b2 = fig.add_subplot(b_gs[1, 0])
    traits_b = [("resources", "Money"), ("looks", "Looks"), ("kindness", "Kindness")]
    yb = np.arange(len(traits_b))
    ax_b1.axvline(0, color=MUTED, lw=0.7)
    for (s_k, s_lbl, col, mk), off in zip((("women", "Women", WOMEN, "D"), ("men", "Men", MEN, "o")), (-0.14, 0.14)):
        vals = [kowal["partnering_slopes"]["by_sex"][s_k][tk]["pp_per_sd"] for tk, _ in traits_b]
        los = [kowal["partnering_slopes"]["by_sex"][s_k][tk]["ci_95"][0] for tk, _ in traits_b]
        his = [kowal["partnering_slopes"]["by_sex"][s_k][tk]["ci_95"][1] for tk, _ in traits_b]
        ax_b1.errorbar(
            vals, yb + off, xerr=[[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]],
            color=col, fmt=mk, ms=3.8, elinewidth=0.95, capsize=1.8, label=s_lbl,
        )
    ax_b1.grid(False)
    ax_b1.grid(axis="x", color=GRID, lw=0.5)
    ax_b1.set_yticks(yb, [t[1] for t in traits_b], fontsize=6.4)
    ax_b1.invert_yaxis()
    ax_b1.set_xlim(-1.4, 6.4)
    ax_b1.set_xticks([0, 2, 4, 6])
    n_door_k = kowal["partnering_slopes"]["n"] / 1000.0
    ax_b1.set_xlabel(f"1. Secondary entry gate (pp / SD, {n_door_k:.1f}k)", fontsize=6.2, labelpad=1.8)
    panel(ax_b1, "B", "Secondary entry vs. primary bond", fontsize=7.6, pad=4)
    aud_r = hcm["audit_compression_ratios"]["institutional_kin_to_app_mean_ratio"]
    ax_b1.text(
        6.28, 1.90,
        f"{pvc_gap:.0f}% photo blind;\nApp audit \u00f7{aud_r:.1f}",
        fontsize=4.85, color=INK2, ha="right", va="center", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )

    ax_b2.axvline(0, color=MUTED, lw=0.7)
    prqc_s = kowal["pair_bond_reversal"]["relationship_quality_prqc"]["by_sex"]
    for (s_k, s_lbl, col, mk), off in zip((("women", "Women", WOMEN, "D"), ("men", "Men", MEN, "o")), (-0.14, 0.14)):
        vals = [prqc_s[s_k]["traits"][tk]["sd_per_sd"] for tk, _ in traits_b]
        los = [prqc_s[s_k]["traits"][tk]["ci_95"][0] for tk, _ in traits_b]
        his = [prqc_s[s_k]["traits"][tk]["ci_95"][1] for tk, _ in traits_b]
        ax_b2.errorbar(
            vals, yb + off, xerr=[[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]],
            color=col, fmt=mk, ms=3.8, elinewidth=0.95, capsize=1.8, label=s_lbl,
        )
    ax_b2.grid(False)
    ax_b2.grid(axis="x", color=GRID, lw=0.5)
    ax_b2.set_yticks(yb, [t[1] for t in traits_b], fontsize=6.4)
    ax_b2.invert_yaxis()
    ax_b2.set_xlim(-0.04, 0.39)
    ax_b2.set_xticks([0.0, 0.15, 0.30])
    n_bond_k = kowal["pair_bond_reversal"]["relationship_quality_prqc"]["n"] / 1000.0
    ax_b2.set_xlabel(f"2. Primary bond love (SD / SD, {n_bond_k:.1f}k)", fontsize=6.2, labelpad=2.5)
    ax_b2.legend(loc="lower left", fontsize=5.4, handletextpad=0.25, labelspacing=0.15)
    k_fit_r = chk_kow["partner_kindness"]["estimate"] / chk_kow["idiographic_checklist_fit"]["estimate"]
    ax_b2.text(
        0.382, -0.18,
        f"Money & marry-up \u2248 0;\nKindness {k_fit_r:.1f}\u00d7 checklist",
        fontsize=4.85, color=INK, ha="right", va="top", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88),
    )

    # Panel C: Age-41 Horizon Crossover vs. Unbroken Devotion & 94% Pairing Gate
    ax_c = fig.add_subplot(outer[0, 2])
    wins = ("18-24", "25-29", "30-39", "40-49", "50-59", "60-69")
    wx = np.array([21.0, 27.0, 34.5, 44.5, 54.5, 64.5])
    w_ser = np.array([100.0 * pov["market_and_commitment"][w]["by_sex"]["Female"]["serious_commitment"]["share"] for w in wins])
    m_ser = np.array([100.0 * pov["market_and_commitment"][w]["by_sex"]["Male"]["serious_commitment"]["share"] for w in wins])
    ax_c.fill_between(wx, m_ser, w_ser, color=WOMEN, alpha=0.13, lw=0)
    ax_c.plot(wx, w_ser, color=WOMEN, lw=1.6, marker="D", ms=3.1, label="Women")
    ax_c.plot(wx, m_ser, color=MEN, lw=1.6, marker="o", ms=3.1, label="Men")

    p_bands = [
        (lo, hi) for lo, hi in data["cohorts"]["pov_bands"]
        if hi <= 69
        and pov["bands"][f"{lo}-{hi}"]["Female"]["without_children"]
        and pov["bands"][f"{lo}-{hi}"]["Male"]["without_children"]
    ]
    bx = np.array([0.5 * (lo + hi) for lo, hi in p_bands])
    bk = [f"{lo}-{hi}" for lo, hi in p_bands]
    w_k = np.array([100.0 * pov["bands"][b]["Female"]["without_children"]["shares"]["want"] for b in bk])
    m_k = np.array([100.0 * pov["bands"][b]["Male"]["without_children"]["shares"]["want"] for b in bk])
    ax_c.fill_between(bx, m_k, w_k, where=(w_k >= m_k), interpolate=True, color=WOMEN, alpha=0.12, lw=0)
    ax_c.fill_between(bx, m_k, w_k, where=(m_k >= w_k), interpolate=True, color=MEN, alpha=0.14, lw=0)
    ax_c.plot(bx, w_k, color=WOMEN, lw=1.35, ls=(0, (3, 1.8)))
    ax_c.plot(bx, m_k, color=MEN, lw=1.35, ls=(0, (3, 1.8)))

    x_cross = pov["childless_crossover"]["band_midpoint_interpolated_age"]
    ax_c.axvline(x_cross, color=VIOLET, lw=0.85, ls=":", zorder=2)
    ax_c.scatter([x_cross], [np.interp(x_cross, bx, w_k)], s=22, facecolor=VIOLET, edgecolor="white", lw=0.5, zorder=6)
    ax_c.text(19.5, 86.5, "Serious commitment: W > M always", fontsize=5.3, color=INK, fontweight="bold")
    ax_c.text(x_cross + 1.4, 39.5, f"Age ~{x_cross:.0f}\nwant-kids\ncrossover", fontsize=5.1, color=VIOLET, fontweight="bold", linespacing=1.02)
    ax_c.text(
        19.5, 4.5, "U.S. birth drop: 94% = non-marriage\n(observational; ~2.7 ideal unchanged)",
        fontsize=4.9, color=INK2, linespacing=1.04,
        bbox=dict(boxstyle="round,pad=0.15", fc=BAND, ec="none", alpha=0.92),
    )
    ax_c.set_xlim(18, 69)
    ax_c.set_ylim(0, 96)
    ax_c.set_xticks([20, 35, 50, 65])
    ax_c.set_yticks([0, 30, 60, 90])
    panel(ax_c, "C", "Clocks vs. devotion (1.14M)", xlabel="Age (years)", ylabel="Share of profiles (%)", fontsize=7.6, pad=4)
    ax_c.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=5.5, handlelength=1.4, labelspacing=0.18)

    return pdf(fig)


def _v9_json(name):
    return json.loads((V9_ROOT / "results" / f"{name}.json").read_text())


def _load_fossils_v9():
    code_dir = str(V9_ROOT / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import load_fossils  # noqa: PLC0415
    return load_fossils.gingerich(), load_fossils.will_full()


# ---------------------------------------------------------------------------
# 1B. DEEP-TIME FOSSILS, LR04 PALEOCLIMATE, CHILDHOOD & NHS CAMPFIRE (Fig. 2)
# ---------------------------------------------------------------------------
def fossils_paleoclimate_plot(data):
    pp = data["paleoclimate_plateau"]
    osyn = data["open_synthesis"]
    fossils = osyn["v9_anchors"]["fossils"]
    cc = osyn["childhood_coevolution"]
    nhs = osyn["nhs_campfire"]
    e13 = _v9_json("e13_fossils")
    gi_rows, will_rows = _load_fossils_v9()

    fig = plt.figure(figsize=(7.24, 5.48))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.22, 1.04, 1.08], height_ratios=[1.34, 1.0], wspace=0.35, hspace=0.47)
    fig.subplots_adjust(left=0.076, right=0.988, bottom=0.095, top=0.930)

    # Panel A: 2-tier synchronized Cranial Stasis-Step + LR04 Benthic d18O Stack
    a_gs = outer[0, 0].subgridspec(2, 1, height_ratios=[1.75, 1.0], hspace=0.08)
    ax_top = fig.add_subplot(a_gs[0, 0])
    ax_lr = fig.add_subplot(a_gs[1, 0], sharex=ax_top)

    for a in (ax_top, ax_lr):
        a.axvspan(1.50, 0.70, color="#ECE8DD", zorder=0, lw=0)
        a.axvspan(0.93, 0.813, color=VIOLET, alpha=0.12, zorder=1, lw=0)
        a.axvline(1.12, color=MEN, lw=0.8, ls=":", alpha=0.85, zorder=2)
        a.axvline(0.789, color=GOLD, lw=0.75, ls="--", alpha=0.85, zorder=2)

    gi_age = np.array([g[2] for g in gi_rows], dtype=float)
    gi_ecv = np.array([g[3] for g in gi_rows], dtype=float)
    ax_top.scatter(gi_age, gi_ecv, s=7, facecolor=GREY, edgecolor="none", alpha=0.32, zorder=2)

    gstyle = {
        "aus": (GREY, "o", "Australopiths"),
        "eh": (B300, "s", "Early Homo"),
        "mp": (B550, "D", "Mid-Pleist."),
        "nea": (WOMEN, "^", "Neanderthals"),
        "sap": (AQUA, "o", "H. sapiens"),
    }
    for k in ("aus", "eh", "mp", "sap", "nea"):
        pts = np.asarray(fossils["groups"][k], dtype=float)
        c, m, lbl = gstyle[k]
        dot(ax_top, pts[:, 0], pts[:, 1], c, m, s=10 if m != "D" else 8, z=4, label=lbl)

    xx = np.linspace(3.25, 0.01, 600)
    lin_y = np.exp(np.log(gi_ecv).mean() + e13["slope_specimen"] * (-xx + gi_age.mean()))
    ax_top.plot(xx, lin_y, color=MUTED, lw=0.95, ls=(0, (3, 2)), zorder=3, label="Log-linear")
    ax_top.plot(xx, fossil_step_curve(fossils, xx), color=INK, lw=1.65, zorder=5, label="Stasis\u2013step")

    ax_top.annotate(
        "", xy=(1.12, 1520), xytext=(1.50, 1520),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.75),
    )
    ax_top.text(1.31, 1640, "380\u2013711 kyr lead", fontsize=4.85, color=INK, ha="center", va="bottom", fontweight="bold")
    ax_top.text(
        1.14, 1230, "Stalled 800 kyr\n(~908 cm\u00b3)", fontsize=4.85, color=INK, ha="center", va="center", fontweight="bold", linespacing=1.0,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.92),
    )
    for xf in (1.0, 0.78):
        ax_top.plot([xf], [315], marker="^", ms=3.5, color=AQUA, zorder=6)
    ax_top.text(
        0.88, 405, "fire & cooking", fontsize=4.7, color=AQUA, ha="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.82),
    )

    ax_top.set_yscale("log")
    ax_top.set_ylim(285, 3150)
    ax_top.set_yticks([400, 700, 1200, 2200])
    ax_top.set_yticklabels(["400", "700", "1,200", "2,200"])
    ax_top.minorticks_off()
    plt.setp(ax_top.get_xticklabels(), visible=False)
    panel(ax_top, "A", "Growth, stasis & LR04 stack", ylabel="cm$^3$ (log)", fontsize=7.6)
    ax_top.legend(loc="upper left", fontsize=4.45, ncol=2, columnspacing=0.40, labelspacing=0.12, handletextpad=0.18)

    lr = pp["lr04_plot_series"]
    lr_age = np.asarray(lr["age_ma"], dtype=float)
    lr_d18 = np.asarray(lr["d18o_permil"], dtype=float)
    env_age = np.asarray(lr["envelope_age_ma"], dtype=float)
    ax_lr.fill_between(env_age, lr["envelope_p10"], lr["envelope_p90"], color=B300, alpha=0.22, lw=0)
    ax_lr.plot(lr_age, lr_d18, color=MEN, lw=0.6, alpha=0.88, zorder=3)
    ax_lr.plot(env_age, lr["envelope_mean"], color=INK, lw=1.05, zorder=4)
    ax_lr.set_xlim(3.32, -0.06)
    ax_lr.set_ylim(2.85, 5.25)
    ax_lr.set_yticks([3.2, 4.0, 4.8])
    ax_lr.set_xticks([3, 2, 1, 0])
    ax_lr.grid(axis="y", color=GRID, lw=0.5)
    ax_lr.set_xlabel("Million years ago (Ma)", labelpad=2.5)
    ax_lr.set_ylabel("LR04 $\\delta^{18}$O (\u2030)", fontsize=6.2, labelpad=2.5)
    ax_lr.text(2.25, 4.82, "41-kyr", fontsize=4.7, color=INK2, ha="center")
    ax_lr.text(1.13, 4.92, "MIS 34", fontsize=4.7, color=MEN, ha="right", fontweight="bold")
    ax_lr.text(0.35, 4.95, "100-kyr MPT", fontsize=4.7, color=INK, ha="center")

    # Panel B: 10-Event Chronological & Epistemic Tier Timeline
    ax_b = fig.add_subplot(outer[0, 1])
    ax_b.axvspan(1.50, 0.70, color="#ECE8DD", zorder=0, lw=0)
    ax_b.axvline(1.50, color=INK, lw=0.8, ls="--", zorder=2)
    tier_col = {
        "measured_climate": MEN,
        "contested_genomic": VIOLET,
        "unlinked_physical": GOLD,
        "energetic_release": AQUA,
    }
    tier_mk = {
        "measured_climate": "s",
        "contested_genomic": "D",
        "unlinked_physical": "o",
        "energetic_release": "^",
    }
    ev_labels = {
        "mpt_intensification": "MPT 41\u2192100 kyr",
        "mis34_europe_cooling": "MIS 34 Eur. depop.",
        "bosumtwi_impact": "Bosumtwi (10.5 km)",
        "wonderwerk_fire": "Wonderwerk fire",
        "fitcoal_bottleneck": "Ne\u22481,280 (contested)",
        "pantasma_impact": "Pantasma (14 km)",
        "oldest_toba_tuff": "Oldest Toba",
        "australasian_tektite_impact": "Bolaven tektites",
        "gby_cooking": "GBY fish cooking",
        "matuyama_brunhes": "Matuyama\u2013Brunhes",
    }
    evs = pp["candidate_events"]
    for idx_e, ev in enumerate(evs):
        tc = tier_col[ev["tier"]]
        mk = tier_mk[ev["tier"]]
        w_lo, w_hi = ev["window_ma"]
        ax_b.plot([w_hi, w_lo], [idx_e, idx_e], color=tc, lw=2.1, solid_capstyle="round", alpha=0.80, zorder=3)
        dot(ax_b, [ev["point_ma"]], [idx_e], tc, mk, s=15, z=4)
    ax_b.set_yticks(range(len(evs)), [ev_labels.get(e["id"], e["short_label"]) for e in evs], fontsize=5.2)
    ax_b.invert_yaxis()
    ax_b.set_xlim(1.62, 0.58)
    ax_b.grid(False)
    ax_b.grid(axis="x", color=GRID, lw=0.55)
    panel(ax_b, "B", "Plateau crisis & release audit", xlabel="Age window (Ma)", fontsize=7.6)
    du_sl = pp["du2018_age_uncertainty_mc"]["slopes"]
    s_pre = du_sl["phase_II_2.0_1.5"]["median_slope_ln_per_myr"]
    s_plt = du_sl["phase_III_1.5_0.7"]["median_slope_ln_per_myr"]
    ax_b.text(
        1.60, 8.95,
        f"Du 2018: {s_pre:.2f}\u2192{s_plt:.2f}\n(\u221258% slowdown)",
        fontsize=4.7, color=INK, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=GRID, lw=0.45, alpha=0.92),
    )
    leg_b = [
        Line2D([], [], color=MEN, marker="s", lw=1.4, ms=3.0, label="Climate"),
        Line2D([], [], color=VIOLET, marker="D", lw=1.4, ms=3.0, label="Contested Ne"),
        Line2D([], [], color=GOLD, marker="o", lw=1.4, ms=3.0, label="Unlinked impact"),
        Line2D([], [], color=AQUA, marker="^", lw=1.4, ms=3.0, label="Fire / cooking"),
    ]
    ax_b.legend(
        handles=leg_b, loc="upper left", bbox_to_anchor=(0.06, 0.72), fontsize=4.6,
        labelspacing=0.14, handlelength=1.1, frameon=True, facecolor="white", framealpha=0.88, edgecolor="none",
    )

    # Panel C: Childhood 66.3% Brain Peak & Proposition 22 (E22) Pair-Bond Encephalization Ceiling
    c_gs = outer[0, 2].subgridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.48)
    ax_c1 = fig.add_subplot(c_gs[0, 0])
    ax_c2 = fig.add_subplot(c_gs[1, 0])

    traj = cc["kuzawa_2014_brain_metabolism"]["trajectory"]
    t_age = [r["age_years"] for r in traj]
    t_rmr = [r["brain_rmr_pct"] for r in traj]
    t_der = [r["brain_der_pct"] for r in traj]
    ax_c1.fill_between(t_age, t_der, t_rmr, color=WOMEN, alpha=0.13, lw=0)
    ax_c1.plot(t_age, t_rmr, color=WOMEN, lw=1.55, marker="D", ms=2.8, label="% Resting (RMR)")
    ax_c1.plot(t_age, t_der, color=MEN, lw=1.35, ls="--", marker="o", mfc="white", ms=2.7, label="% Daily (DER)")
    ax_c1.axvline(4.5, color=VIOLET, lw=0.75, ls=":")
    sccs_c = cc["sccs_alloparenting_and_sharing"]
    allo_pct = 100.0 * sccs_c["early_childhood_nonmaternal_majority_sccs52"]["share"]
    ax_c1.text(
        5.3, 62.5,
        f"Age 4.5: 66.3% RMR\nSCCS: 0% solo mom;\n{allo_pct:.1f}% allomaternal",
        fontsize=4.65, color=INK, va="top", ha="left", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    ax_c1.set_xlim(-0.5, 22.8)
    ax_c1.set_ylim(0, 76)
    ax_c1.set_xticks([0, 4.5, 10, 16, 22], ["0", "4.5", "10", "16", "22"])
    ax_c1.set_xlabel("Childhood age (yr, Kuzawa 2014)", fontsize=5.9, labelpad=1.5)
    ax_c1.set_ylabel("Brain %", fontsize=6.0)
    panel(ax_c1, "C", "Child brain peak & love ceiling", fontsize=7.6, pad=4)
    ax_c1.legend(loc="lower left", fontsize=4.55, labelspacing=0.12, handlelength=1.15, frameon=True, facecolor="white", framealpha=0.90, edgecolor="none")

    e22_regs = cc["proposition_22_e22_coevolution"]["regimes"]
    r_specs = [
        ("solitary_ape_mother", "Solitary ape\nmother", GREY),
        ("early_homo_allomaternal_raw", "Raw food +\nalloparents", B300),
        ("cooking_without_pair_bond_love", "Cooking, no\npair-bond", GOLD),
        ("late_homo_cooking_plus_pair_bond_love", "Cooking +\npair-bond love", AQUA),
    ]
    yc2 = np.arange(len(r_specs))
    for i, (rk, _, rcol) in enumerate(r_specs):
        cf_v = e22_regs[rk]["equilibrium_cm3_closed_form"]
        sm_v = e22_regs[rk]["equilibrium_cm3_simulated"]
        ax_c2.barh(i, cf_v, height=0.58, color=rcol, alpha=0.88, zorder=2)
        dot(ax_c2, [sm_v], [i], INK, "D", s=12, z=4)
        ax_c2.text(cf_v + 24, i, f"{cf_v:,.0f}", va="center", ha="left", fontsize=5.1, fontweight="bold", color=INK)
    ax_c2.grid(False)
    ax_c2.grid(axis="x", color=GRID, lw=0.5)
    ax_c2.set_yticks(yc2, [r[1] for r in r_specs], fontsize=5.0)
    ax_c2.invert_yaxis()
    ax_c2.set_xlim(0, 1620)
    ax_c2.set_xticks([0, 650, 1300], ["0", "650", "1,300"])
    love_g = cc["proposition_22_e22_coevolution"]["love_devotion_gain_cm3"]
    ax_c2.set_xlabel(f"Prop. 22 E22 equilibrium cm\u00b3 (+{love_g:.0f} love)", fontsize=5.8, labelpad=2.0)

    # Panel D: Three Post-0.7 Ma Parallel Regional Series (Will et al. 2021)
    ax_d = fig.add_subplot(outer[1, 0])
    ser_specs = [
        ("africa_to_sapiens", AQUA, "o", "Africa \u2192 H. sapiens",
         lambda w: (w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "Africa") or w["taxon"] == "Pleistocene Homo sapiens"),
        ("europe_w_asia", WOMEN, "^", "W Eurasia \u2192 Neand.",
         lambda w: w["region"] == "Europe/W Asia" and w["taxon"] != "Pleistocene Homo sapiens"),
        ("east_asia_archaic", MEN, "D", "E/SE Asia archaic",
         lambda w: w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "E/SE Asia"),
    ]
    for key, col, mk, lab, pred in ser_specs:
        wa = np.array([w["age"] for w in will_rows if pred(w)], dtype=float)
        we = np.array([w["ecv"] for w in will_rows if pred(w)], dtype=float)
        dot(ax_d, wa, we, col, mk, s=10, alpha=0.85)
        sl = e13["series"][key]["slope"]
        x2 = np.linspace(wa.max(), wa.min(), 50)
        ax_d.plot(x2, np.exp(np.log(we).mean() + sl * (-x2 + wa.mean())), color=col, lw=1.55, label=f"{lab} ({sl:.2f})")
    ax_d.set_yscale("log")
    ax_d.set_xlim(0.95, -0.02)
    ax_d.set_ylim(780, 1980)
    ax_d.set_yticks([800, 1000, 1200, 1500, 1800])
    ax_d.set_yticklabels(["800", "1,000", "1,200", "1,500", "1,800"])
    ax_d.minorticks_off()
    panel(ax_d, "D", "Post-0.7 Ma regional surge", xlabel="Million years ago (Ma)", ylabel="Volume (cm$^3$)", fontsize=7.6)
    ax_d.legend(loc="upper left", fontsize=5.1, labelspacing=0.18)

    # Panel E: Late Homo Coexistence of Large- and Small-Brained Lineages
    ax_e = fig.add_subplot(outer[1, 1])
    rng_d = np.random.default_rng(3)
    order_d = [("mp", "Mid-Pleist.\nHomo", B550), ("nea", "Neander-\nthals", WOMEN), ("sap", "Pleist.\nH. sapiens", AQUA)]
    for i, (gk, glab, gcol) in enumerate(order_d):
        vals = np.asarray(fossils["groups"][gk], dtype=float)[:, 1]
        ax_e.scatter(i + rng_d.uniform(-0.17, 0.17, len(vals)), vals, s=7.5, facecolor=gcol, edgecolor="white", linewidth=0.3, alpha=0.82, zorder=3)
        ax_e.plot([i - 0.26, i + 0.26], [vals.mean(), vals.mean()], color=INK, lw=1.45, zorder=4)
        ax_e.text(i, vals.max() + 45, f"{vals.mean():,.0f}", ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=INK)
    ax_e.axhline(1349, color=MUTED, lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax_e.text(
        2.48, 1295, "recent 1,349", ha="right", va="top", fontsize=4.75, color=INK2,
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.88),
    )
    ax_e.axhspan(465, 560, color=VIOLET, alpha=0.16, lw=0, zorder=1)
    ax_e.text(1.0, 512, "H. naledi (465\u2013560)", ha="center", va="center", fontsize=5.1, color=VIOLET, fontweight="bold")
    ax_e.axhline(426, color=GOLD, lw=1.05, zorder=2)
    ax_e.text(1.0, 392, "H. floresiensis (426)", ha="center", va="top", fontsize=5.1, color=GOLD, fontweight="bold")
    ax_e.set_xticks(range(3), [o[1] for o in order_d], fontsize=5.8)
    ax_e.set_xlim(-0.5, 2.55)
    ax_e.set_ylim(280, 2020)
    ax_e.set_yticks([400, 800, 1200, 1600, 2000])
    ax_e.set_yticklabels(["400", "800", "1,200", "1,600", "2,000"])
    panel(ax_e, "E", "Late Homo & insular dwarfs", ylabel="Volume (cm$^3$)", fontsize=7.6)

    # Panel F: Harvard NHS 86-Society Campfire Acoustic Legibility Gradient (Mehr et al. 2019)
    ax_f = fig.add_subplot(outer[1, 2])
    alg = nhs["acoustic_legibility_gradient"]
    nhs_rows = [
        ("lullaby", "Infant lullaby", AQUA),
        ("dance", "Communal dance", MEN),
        ("healing", "Ceremonial healing", B300),
        ("love", "Romantic love song", GOLD),
    ]
    yf = np.arange(len(nhs_rows))
    ax_f.axvline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    for i, (nk, _, ncol) in enumerate(nhs_rows):
        d_val = alg[nk]["cohen_d"]
        d_lo, d_hi = alg[nk]["cohen_d_ci_95"]
        au = alg[nk]["auroc"]
        ax_f.barh(i, d_val, height=0.52, color=ncol, alpha=0.86, zorder=2)
        ax_f.errorbar(d_val, i, xerr=[[d_val - d_lo], [d_hi - d_val]], color=INK, fmt="none", elinewidth=0.95, capsize=2.0, zorder=4)
        ax_f.text(d_hi + 0.06, i, f"+{d_val:.2f}\n(AUC {au:.2f})", va="center", ha="left", fontsize=4.65, fontweight="bold", color=INK, linespacing=0.96)
    ax_f.grid(False)
    ax_f.grid(axis="x", color=GRID, lw=0.5)
    ax_f.set_yticks(yf, [r[1] for r in nhs_rows], fontsize=5.4)
    ax_f.invert_yaxis()
    ax_f.set_xlim(-0.22, 3.42)
    ax_f.set_xticks([0.0, 1.0, 2.0, 3.0])
    d_sh = 100.0 * nhs["ethnographic_hearth_contexts"]["any_communal_dance"]["share_of_coded"]
    ll_r = nhs["lullaby_to_love_d_ratio"]
    ax_f.text(
        3.35, 3.42,
        f"Lullaby {ll_r:.1f}\u00d7 love d;\n{d_sh:.1f}% hearth dance",
        fontsize=4.7, color=INK, ha="right", va="bottom", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_f, "F", "Campfire song legibility (86 soc.)", xlabel="Blind listener Cohen's $d$ (29.4k raters)", fontsize=7.6)

    return pdf(fig)


# ---------------------------------------------------------------------------
# 2. UNIFIED SPEED-DATING, HOBBY NULL & HINDSIGHT FIGURE (performance-fig-jury-choice.pdf)
# ---------------------------------------------------------------------------
def jury_choice_plot(data):
    speed = data["speed_dating"]
    tw = data["open_synthesis"]["two_way_jury"]
    chk = speed["checklist_and_hindsight"]
    hw = chk["hobby_checklist_vs_live_wavelength"]
    wps = chk["within_person_shifts"]
    fup = chk["post_event_followup_3_4_weeks"]
    e17 = _v9_json("e17_speed")

    fig, axes = plt.subplots(2, 3, figsize=(7.24, 4.95))
    fig.subplots_adjust(left=0.088, right=0.988, bottom=0.105, top=0.905, wspace=0.38, hspace=0.55)

    # Panel A: Stated vs Revealed Choice Weights (Said -> Did as Shares of 100)
    ax = axes[0, 0]
    rows_a = ["attr", "shar", "fun", "intel", "sinc", "amb"]
    lbls_a = ["Looks", "Shared\ninterests", "Fun", "Intelli-\ngence", "Sincerity", "Ambition"]
    for i, a in enumerate(rows_a):
        y0 = len(rows_a) - 1 - i
        for n_sex, col, mk, dy in (("women", WOMEN, "D", -0.16), ("men", MEN, "o", 0.16)):
            pos = {b: max(e17["did"]["own"][n_sex]["coef"][b][0], 0.0) for b in rows_a}
            did = 100.0 * pos[a] / sum(pos.values())
            said = e17["said"][n_sex][a]
            ax.annotate(
                "", xy=(did, y0 + dy), xytext=(said, y0 + dy),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=0.95, mutation_scale=5.5, shrinkA=1.8, shrinkB=2.2),
                zorder=2,
            )
            ax.scatter([said], [y0 + dy], s=14 if mk == "o" else 11, marker=mk, facecolor="white", edgecolor=col, linewidth=0.95, zorder=3)
            ax.scatter([did], [y0 + dy], s=17 if mk == "o" else 13, marker=mk, facecolor=col, edgecolor="white", linewidth=0.45, zorder=4)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(range(len(rows_a)), lbls_a[::-1], fontsize=5.8)
    ax.set_ylim(-0.6, len(rows_a) - 0.4)
    ax.set_xlim(-2, 56)
    panel(ax, "A", "Stated vs. revealed (said \u2192 did)", xlabel="Share of positive weight (%)", fontsize=7.6)
    leg_a = [
        Line2D([], [], marker="D", ls="", ms=3.5, mfc=WOMEN, mec="white", label="Women"),
        Line2D([], [], marker="o", ls="", ms=3.7, mfc=MEN, mec="white", label="Men"),
        Line2D([], [], marker="o", ls="", ms=3.5, mfc="white", mec=INK2, label="Stated"),
        Line2D([], [], marker="o", ls="", ms=3.7, mfc=INK2, mec="white", label="Revealed"),
    ]
    ax.legend(handles=leg_a, loc="lower right", fontsize=5.1, ncol=2, columnspacing=0.45)

    # Panel B: 4-Minute Inter-Observer Agreement (r1) vs Panel Choice Slope
    ax = axes[0, 1]
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    r1_map = {a: e17["jury"][a]["r1"] for a in ("attr", "fun", "intel", "amb", "sinc")}
    for n_sex, col, mk, lab in (("women", WOMEN, "D", "Women"), ("men", MEN, "o", "Men")):
        for idx_a, a in enumerate(r1_map):
            c = e17["did"]["panel"][n_sex]["coef"][a]
            ax.vlines(r1_map[a], c[1], c[2], color=col, lw=1.0, zorder=2)
            dot(ax, [r1_map[a]], [c[0]], col, mk, s=18 if mk == "o" else 14, z=3, label=lab if idx_a == 0 else None)
    for x_t, y_t, txt in ((0.385, 8.2, "Looks"), (0.265, 9.6, "Fun"), (0.155, 7.6, "Intel.,\nambition"), (0.088, -6.3, "Sincerity")):
        ax.text(x_t, y_t, txt, fontsize=5.3, color=INK2, ha="center", va="center", linespacing=1.02)
    ax.set_xlim(0.0, 0.45)
    ax.set_ylim(-9.0, 21.5)
    panel(ax, "B", "4-min legibility governs choice", xlabel="Single-observer agreement ($r_1$)", ylabel="Yes slope / panel SD (pp)", fontsize=7.6)
    ax.legend(loc="upper left", fontsize=5.5)

    # Panel C: Two-Way Observer x Context Validity Law (Proposition 20)
    ax = axes[0, 2]
    m_grid = [1, 2, 4, 8, 16, 32, 64]
    styles = [
        ("1", GOLD, "-", "o", "$k=1$ (1 photo/date)"),
        ("2", B300, "--", "s", "$k=2$ contexts"),
        ("4", MEN, "-.", "^", "$k=4$ contexts"),
        ("8", AQUA, "-", "D", "$k=8$ (camp)"),
        ("32", VIOLET, ":", "v", "$k=32$ contexts"),
    ]
    for k_str, col, ls, mk, lab in styles:
        ys = [tw["surface_r2"][k_str][str(m)] for m in m_grid]
        ax.plot(m_grid, ys, color=col, ls=ls, lw=1.4, marker=mk, ms=3.0, label=lab)
    ax.axhline(tw["single_context_asymptote_r2"], color=GOLD, lw=0.7, ls=(0, (2, 2)))
    ax.axhline(tw["multi_context_halo_ceiling_r2"], color=VIOLET, lw=0.7, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xticks(m_grid, [str(m) for m in m_grid], fontsize=5.8)
    ax.minorticks_off()
    ax.set_ylim(0.11, 0.82)
    panel(ax, "C", "More raters \u2260 more contexts (Prop. 20)", xlabel="Independent observers ($m$)", ylabel="Validity $R^2_{m,k}$", fontsize=7.6)
    ax.legend(loc="lower right", fontsize=4.9, labelspacing=0.16)

    # Panel D: Unique Dyadic Spark vs. Consensus Trophy Value (Unbiased Bipartite SRM)
    ax = axes[1, 0]
    dc = speed["dyadic_chemistry"]
    vp = dc["variance_partition"]
    items_d = [
        ("attr", "Looks"),
        ("like", "Romantic liking\n(4,011 dyads)"),
        ("shar", "Shared interests"),
        ("intel", "Intelligence"),
        ("sinc", "Sincerity"),
    ]
    segs_d = [
        ("partner_consensus", "Consensus", GOLD),
        ("chooser_leniency", "Leniency", B300),
        ("unique_dyadic_and_noise", "Unique spark", VIOLET),
    ]
    for i, (tk, tlab) in enumerate(items_d):
        left = 0.0
        sh = vp[tk]["unbiased_srm_shares"]
        for sk, _, col in segs_d:
            v = 100.0 * sh[sk]
            ax.barh(i, max(v - 0.5, 0.0), left=left + 0.25, height=0.56, color=col, zorder=2)
            if v >= 14.0:
                ax.text(left + v / 2.0, i, f"{v:.0f}%", ha="center", va="center", fontsize=5.1, color="white" if col == VIOLET else INK, fontweight="bold")
            left += v
    ax.grid(False)
    ax.set_yticks(range(len(items_d)), [x[1] for x in items_d], fontsize=5.5)
    ax.set_ylim(-0.55, 5.10)
    ax.invert_yaxis()
    ax.set_xlim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    rec_r = dc["mutual_dyadic_reciprocity_like"]["estimate"]
    panel(ax, "D", f"Dyadic spark beats consensus (r={rec_r:+.2f})", xlabel="Unbiased SRM share (%)", pad=16, fontsize=7.6)
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in segs_d],
        loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=4.6,
        handlelength=0.65, handletextpad=0.22, columnspacing=0.45,
    )

    # Panel E: Objective 17-Hobby Profile Tag Null vs. Live Conversational Wavelength (7,116 Dyads)
    ax = axes[1, 1]
    fe_items = [
        ("int_corr", "17-hobby profile\ntag corr."),
        ("sinc", "Sincerity"),
        ("intel", "Intelligence"),
        ("fun", "Live fun"),
        ("shar", "Live shared\nwavelength"),
        ("attr", "Live looks"),
    ]
    ye = np.arange(len(fe_items))
    ax.axvline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    for (sx, slb, col, mk), off in zip((("women", "Women", WOMEN, "D"), ("men", "Men", MEN, "o")), (-0.14, 0.14)):
        cfs = hw["chooser_fe_slopes"][sx]["coefficients_pp"]
        ests = [cfs[k]["estimate"] for k, _ in fe_items]
        los = [cfs[k]["interval_95"][0] for k, _ in fe_items]
        his = [cfs[k]["interval_95"][1] for k, _ in fe_items]
        ax.errorbar(
            ests, ye + off, xerr=[[e - l for e, l in zip(ests, los)], [h - e for e, h in zip(ests, his)]],
            color=col, fmt=mk, ms=3.5, elinewidth=0.95, capsize=1.8, label=slb,
        )
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(ye, [x[1] for x in fe_items], fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(-4.5, 25.5)
    r_hob = hw["r_objective_hobby_corr_yes"]["estimate"]
    r_wav = hw["r_live_shared_wavelength_yes"]["estimate"]
    rat_hw = hw["live_to_checklist_r_ratio"]
    ax.text(
        24.8, 0.15,
        f"7,116 dyads:\nHobby r={r_hob:+.3f} (null)\nPlay r={r_wav:+.3f} ({rat_hw:.1f}\u00d7)",
        fontsize=4.75, color=INK, ha="right", va="top", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.14", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax, "E", "17-hobby tag null vs. live play", xlabel="Chooser-FE Yes slope (pp / SD)", fontsize=7.6)

    # Panel F: Longitudinal Waves 1-3 Hindsight Awakening (24h Looks Hijack -> 3-Week Intel Snapback)
    ax = axes[1, 2]
    ax.axhline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    sh_24_w = wps["nextday_realized_minus_pre_stated"]["women"]["delta_points"]
    sh_24_m = wps["nextday_realized_minus_pre_stated"]["men"]["delta_points"]
    sh_3w_w = wps["threeweek_stated_minus_nextday_realized"]["women"]["delta_points"]
    sh_3w_m = wps["threeweek_stated_minus_nextday_realized"]["men"]["delta_points"]
    cats_f = [
        ("24h looks\njump", sh_24_w["attr"], sh_24_m["attr"]),
        ("24h intel.\ndrop", sh_24_w["intel"], sh_24_m["intel"]),
        ("3\u20134 wk looks\nrecede", sh_3w_w["attr"], sh_3w_m["attr"]),
        ("3\u20134 wk intel.\nsnapback", sh_3w_w["intel"], sh_3w_m["intel"]),
    ]
    xf = np.arange(len(cats_f))
    for i, (_, w_cell, m_cell) in enumerate(cats_f):
        for cell, col, mk, off, lb in ((w_cell, WOMEN, "D", -0.14, "Women"), (m_cell, MEN, "o", 0.14, "Men")):
            est = cell["estimate"]
            lo, hi = cell["interval_95"]
            ax.errorbar(
                i + off, est, yerr=[[est - lo], [hi - est]],
                color=col, fmt=mk, ms=3.8, elinewidth=0.95, capsize=2.0, label=lb if i == 0 else None,
            )
    ax.set_xticks(xf, [c[0] for c in cats_f], fontsize=5.1)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-8.5, 15.8)
    dt_sh = 100.0 * fup["went_on_followup_date_share_among_matched"]
    cl_sh = 100.0 * fup["called_or_was_called_share_among_matched"]
    ax.text(
        3.42, 14.8,
        f"Matches\u2192real life:\n{cl_sh:.0f}% call, {dt_sh:.0f}% date",
        fontsize=4.75, color=INK, ha="right", va="top", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.14", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax, "F", "Waves 1\u20133 hindsight awakening", ylabel="\u0394 weight (pts of 100)", fontsize=7.6)
    ax.legend(loc="lower right", fontsize=5.1)

    return pdf(fig)


# ---------------------------------------------------------------------------
# 2B. GENERALIST BREADTH, STAGED CAPACITY, ENFORCED DELIVERY & MIND SELECTION (Fig. 4)
# ---------------------------------------------------------------------------
def generalist_verify_plot(data):
    P = _v9_json("e15_polymath")
    V = _v9_json("e16_verify")
    sat_b = data["speed_dating"]["checklist_and_hindsight"]["objective_sat_blindness"]
    tpp = data["open_synthesis"]["okcupid_jse"]["ten_prompt_portfolio"]
    mdv = tpp["multi_domain_lexical_validity"]
    amr = tpp["assortative_mind_requests_essay9"]

    fig, axes = plt.subplots(2, 3, figsize=(7.24, 4.76))
    fig.subplots_adjust(left=0.072, right=0.988, bottom=0.105, top=0.905, wspace=0.36, hspace=0.49)

    # Panel A: Breadth Buys Pre-Reproductive Time (n*(K) curves across K=1..24)
    ax = axes[0, 0]
    n_ev = np.arange(1, 41)
    k_styles = [("1", GREY), ("3", GOLD), ("6", B300), ("12", MEN), ("24", AQUA)]
    for k_s, col in k_styles:
        yv = P["breadth"]["curves"][k_s]
        ax.plot(n_ev, yv, color=col, lw=1.5, label=f"$K={k_s}$")
    ax.axhline(0.50, color=INK2, lw=0.75, ls=(0, (2, 2)))
    ax.text(36.0, 0.44, "Target $R^* = 0.50$", fontsize=5.3, color=INK2, ha="right")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 40], ["1", "2", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 0.86)
    panel(ax, "A", "Breadth buys time (Prop. 5)", xlabel="Events observed per domain ($n$)", ylabel="General-capacity validity $R_G$", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=5.2, ncol=2, columnspacing=0.5, frameon=True, framealpha=0.90, facecolor="white", edgecolor="none")

    # Panel B: General-Capacity Expenditure Share & 1/K Gaming Dilution (Prop. 6 & Cor. 4)
    ax = axes[0, 1]
    Kc = np.linspace(2, 32, 300)
    for ratio, col, mk in (("4.0", AQUA, "o"), ("8.0", MEN, "s"), ("16.0", VIOLET, "D")):
        rr = float(ratio)
        ax.plot(Kc, Kc / (Kc + rr), color=col, lw=1.5, label=f"$c_G/c_S = {int(rr)}$")
        d_exp = P["expenditure"][ratio]
        dot(ax, [int(k) for k in d_exp], [d_exp[k]["share"][0] for k in d_exp], col, mk, s=14)
    ax.axhline(0.50, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlim(1.8, 35)
    ax.set_ylim(0, 0.96)
    panel(ax, "B", "Generalist share & $1/K$ anti-gaming", xlabel="Common repertoire breadth ($K$ domains)", ylabel="General-capacity share $\\pi_G$", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=5.4)
    ax.text(2.05, 0.83, "Single-domain show\ndiluted as $1/K$ (Cor. 4)", fontsize=5.3, color=INK2, linespacing=1.06)

    # Panel C: Staged Tests Measure Capacity (Rb), Not Unstaged Delivery (Rp) (Prop. 15)
    ax = axes[0, 2]
    vp_par = V["params"]
    vb, sig2, vp, nu, n_occ = vp_par["VB"], vp_par["sig2"], vp_par["VP"], vp_par["nu"], vp_par["n"]
    d_fine = np.linspace(0.0, 1.0, 401)
    rb_curve = [0.0 if d * n_occ <= 0 else vb / (vb + sig2 / (d * n_occ)) for d in d_fine]
    rp_curve = [0.0 if (1 - d) * n_occ <= 0 else vp / (vp + nu / ((1 - d) * n_occ)) for d in d_fine]
    ax.plot(d_fine, rb_curve, color=AQUA, lw=1.6, label="Capacity $R_b(d)$")
    ax.plot(d_fine, rp_curve, color=VIOLET, lw=1.6, label="Delivery $R_p(d)$")
    st_rows = V["staging"]
    dot(ax, [float(k) for k in st_rows], [v["R_b"][0] for v in st_rows.values()], AQUA, "o", s=14)
    dot(ax, [float(k) for k in st_rows], [v["R_p"][0] for v in st_rows.values()], VIOLET, "s", s=13)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1.02)
    panel(ax, "C", f"Staged 'can' vs 'will' ($n={n_occ}$)", xlabel="Share of occasions staged ($d$)", ylabel="Assessment validity", fontsize=7.5)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.43), fontsize=5.4)
    clean_90 = 100.0 * V["clean"]["0.9"]["exact"]
    ax.text(0.50, 0.11, f"At $d=0.9$, 25% defaulter\nclean {clean_90:.0f}% of time", fontsize=5.1, color=INK2, ha="center", linespacing=1.05)

    # Panel D: Third-Party Enforcement Rescues Take-Off & Crowds Out Above Threshold (Prop. 17)
    ax = axes[1, 0]
    E = V["enforcement"]
    eg = np.asarray(E["grid"], dtype=float)
    bmax = E["params"]["bmax"]
    ax.plot(eg, np.asarray(E["b"], dtype=float) / bmax, color=AQUA, lw=1.55, label="Capacity $\\bar b^*/b_{\\max}$")
    ax.plot(eg, E["p"], color=VIOLET, lw=1.55, label="Commitment $\\bar p^*$")
    ax.plot(eg, E["delivered"], color=MEN, lw=1.3, ls="--", label="Delivered ($T=0.6$)")
    ibm = E["ibm"]
    dot(ax, [float(k) for k in ibm], [v["b"][0] / bmax for v in ibm.values()], AQUA, "o", s=14)
    dot(ax, [float(k) for k in ibm], [v["p"][0] for v in ibm.values()], VIOLET, "s", s=13)
    cr = E["crowding"]
    ax.scatter([cr["at"]], [cr["min_delivered"]], s=24, facecolor="white", edgecolor=GOLD, linewidth=1.2, zorder=5, label=f"Crowding ({cr['min_delivered']:.2f})")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.02, 1.05)
    panel(ax, "D", "Enforcement rescues take-off", xlabel="Enforced sharing fraction ($e$)", ylabel="Equilibrium level / share", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=5.0, labelspacing=0.16)

    # Panel E: Empirical Proof of Mind Selection — 4-min SAT Blindness vs. 9-Domain Portfolio
    ax = axes[1, 1]
    ks = np.arange(1, 10)
    sb_vals = [100.0 * mdv["spearman_brown_curve_k_1_to_9"][str(k)] for k in ks]
    ax.axhline(50.0, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.plot(ks, sb_vals, color=AQUA, lw=1.7, marker="o", ms=3.5, label=f"9-prompt lexical $g$ ($n={mdv['n_complete_9_domain_writers']:,}$)")
    sat_1 = 100.0 * sat_b["stranger_intel_sat_r2"]
    sat_p = 100.0 * sat_b["panel_intel_sat_r2"]
    ax.axhline(sat_p, color=GOLD, lw=1.25, ls="--", label=f"10+ dates SAT $R^2$ ($k=1$: {sat_p:.1f}%)")
    ax.scatter([1], [sat_1], color=RED, marker="s", s=26, zorder=5, label=f"1-date SAT $R^2$ ({sat_1:.2f}%)")
    ax.set_xticks(ks)
    ax.set_xlim(0.6, 9.4)
    ax.set_ylim(-2, 70)
    panel(ax, "E", "Live SAT blindness vs 9-domain voice", xlabel="Distinct written domains observed ($K$)", ylabel="Cognitive validity / $R^2$ (%)", fontsize=7.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.99), fontsize=4.85, labelspacing=0.16)
    ax.text(
        9.25, 11.5,
        f"Partner SAT\u2192'Yes':\n$r={sat_b['r_partner_sat_vs_yes']['estimate']:+.3f}$ (null)\n9 domains: {mdv['k9_to_k1_validity_ratio']:.1f}\u00d7 gain",
        fontsize=4.9, color=INK, ha="right", va="bottom", linespacing=1.06,
        bbox=dict(boxstyle="round,pad=0.16", fc=BAND, ec="none", alpha=0.92),
    )

    # Panel F: Assortative Mind Requests Across Candidate Lexical Quintiles (N=44,856)
    ax = axes[1, 2]
    q_keys = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    qx = np.arange(1, 6)
    mind_y = [100.0 * amr["by_lexical_quintile"][q]["ask_mind_curiosity_share"] for q in q_keys]
    mind_lo = [100.0 * amr["by_lexical_quintile"][q]["ask_mind_wilson_ci_95"][0] for q in q_keys]
    mind_hi = [100.0 * amr["by_lexical_quintile"][q]["ask_mind_wilson_ci_95"][1] for q in q_keys]
    kind_y = [100.0 * amr["by_lexical_quintile"][q]["ask_kindness_sincerity_share"] for q in q_keys]
    stat_y = [100.0 * amr["by_lexical_quintile"][q]["ask_height_money_looks_share"] for q in q_keys]
    ax.fill_between(qx, mind_lo, mind_hi, color=VIOLET, alpha=0.15, lw=0)
    ax.plot(qx, mind_y, color=VIOLET, lw=1.65, marker="o", ms=3.5, label="Ask for mind / wit (+40%)")
    ax.plot(qx, kind_y, color=MEN, lw=1.5, marker="s", ms=3.2, label="Ask for kindness (+38%)")
    ax.plot(qx, stat_y, color=GOLD, lw=1.35, ls="--", marker="^", ms=3.2, label="Ask height/money/looks")
    ax.set_xticks(qx, q_keys)
    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(0, 26.2)
    panel(ax, "F", f"Assortative mind requests ($N={amr['n_users_ge3_domains_with_essay9']:,}$)", xlabel="Writer multi-domain lexical quintile", ylabel="'Message me if' share (%)", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=4.9, labelspacing=0.16)
    p0_sh = 100.0 * tpp["prompts"]["essay0"]["share"]
    p8_sh = 100.0 * tpp["prompts"]["essay8"]["share"]
    ax.text(
        5.15, 4.3,
        f"Private admission drops\n{p0_sh:.0f}%\u2192{p8_sh:.1f}% (vulnerability)",
        fontsize=4.8, color=INK2, ha="right", va="bottom", linespacing=1.05,
    )

    return pdf(fig)


# ---------------------------------------------------------------------------
# 3B. ETHNOGRAPHIC JURIES, WAITING WINDOWS & EFFECTIVE ANCESTORS (Fig. 6)
# ---------------------------------------------------------------------------
def ethnography_ancestors_plot(data):
    C = _v9_json("e18_cultures")
    VG = _v9_json("e19_village")
    N = _v9_json("e21_notes")

    fig, axes = plt.subplots(2, 2, figsize=(7.22, 4.56))
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.10, top=0.89, wspace=0.46, hspace=0.50)

    # Panel A: SCCS Spouse-Choice Juries (186 Societies + Foragers)
    ax = axes[0, 0]
    arr = C["arrangement"]
    rows_s = [("Sons", "men"), ("Daughters", "women"), ("Sons, foragers", "men_foragers"), ("Daughters, foragers", "women_foragers")]
    segs = [("alone", "Alone", B300), ("with_approval", "Elders approve", MEN), ("either", "Either", VIOLET), ("parents", "Parents choose", INK)]
    for i, (lab, k) in enumerate(rows_s):
        left = 0.0
        for sk, _, col in segs:
            v = arr[k]["share"][sk]
            ax.barh(i, max(v - 0.6, 0.0), left=left + 0.3, color=col, height=0.60, zorder=2)
            if v >= 11:
                ax.text(left + v / 2.0, i, f"{v:.0f}", ha="center", va="center", fontsize=5.8, color="white" if col != B300 else INK, fontweight="bold")
            left += v
        ax.text(101.5, i, f"n={arr[k]['n']}", fontsize=5.5, color=INK2, va="center")
    ax.grid(False)
    ax.set_yticks(range(4), [r[0] for r in rows_s], fontsize=6.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    panel(ax, "A", "Kin juries decide, firmer for daughters", xlabel="Societies (%)", pad=14)
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, color=c, label=l) for _, l, c in segs],
        loc="lower left", bbox_to_anchor=(-0.18, 1.01), ncol=4, fontsize=5.2,
        handlelength=0.8, handletextpad=0.3, columnspacing=0.6,
    )

    # Panel B: Constraint, Coercion & Divorce Exit
    ax = axes[0, 1]
    cw = C["consent"]["women"]
    rp = C["rape"]
    items = [
        ("Consent non-essential", cw["share"]["2"] + cw["share"]["3"], cw["n"], VIOLET),
        ("Cannot easily object", 100.0 * arr["women_cannot_object"] / arr["women"]["n"], arr["women"]["n"], VIOLET),
        ("Wife-beating present", C["wife_beating"]["share"]["2"], C["wife_beating"]["n"], VIOLET),
        ("Rape reported", rp["sanday"]["share"]["2"], rp["sanday"]["n"], VIOLET),
        ("War captives taken", C["captives"]["share"]["1"], C["captives"]["n"], VIOLET),
        ("Divorce equal access", C["divorce_access"]["share"]["3"], C["divorce_access"]["n"], AQUA),
        ("Divorce common", C["divorce_frequency"]["share"]["common"], C["divorce_frequency"]["n"], AQUA),
    ]
    ys_b = [i + (0.45 if i >= 5 else 0.0) for i in range(len(items))]
    for y_i, (_, v, n_soc, col) in zip(ys_b, items):
        ax.barh(y_i, v, color=col, height=0.58, zorder=2)
        ax.text(v + 1.5, y_i, f"{v:.0f}% ({n_soc})", fontsize=5.6, color=INK2, va="center")
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(ys_b, [it[0] for it in items], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    panel(ax, "B", "Constraint & exit across societies", xlabel="Societies with coded data (%)")

    # Panel C: 171 Binford Forager First-Marriage Ages (Men Wait +6.2 yr)
    ax = axes[1, 0]
    pairs = np.asarray(C["binford_age"]["pairs"], dtype=float)
    ax.plot([8, 36], [8, 36], color=MUTED, lw=0.75, ls=(0, (2, 2)), zorder=1)
    jit = np.random.default_rng(5).normal(0, 0.12, pairs.shape)
    dot(ax, pairs[:, 1] + jit[:, 1], pairs[:, 0] + jit[:, 0], MEN, "o", s=12, alpha=0.82)
    ax.set_xlim(8, 26)
    ax.set_ylim(8, 36)
    panel(ax, "C", f"Men wait +{C['binford_age']['gap']:.1f} yr ({C['binford_age']['n']} forager groups)", xlabel="Women's age at first marriage (yr)", ylabel="Men's age at first marriage (yr)")
    ax.text(25.4, 11.5, f"Men older in {C['binford_age']['men_older_pct']:.0f}%\nof hunter-gatherer societies", fontsize=5.8, color=INK2, ha="right", linespacing=1.1)

    # Panel D: Effective Ancestor Ratio Nef / Nem (Prop. 18)
    ax = axes[1, 1]
    cu = N["ancestors"]["curve"]
    share_m = 100.0 * (1.0 - np.asarray(cu["qm"], dtype=float))
    for a_val, col, lab in ((0.0, B300, "Fathers alike ($\\alpha_m=0$)"), (0.5, MEN, "$\\alpha_m=0.5$"), (1.0, VIOLET, "$\\alpha_m=1.0$"), (2.0, INK, "Unequal ($\\alpha_m=2.0$)")):
        ax.plot(share_m, cu[f"alpha_{a_val}"], color=col, lw=1.45, label=lab)
    ax.axhline(2.0, color=GOLD, lw=0.85, ls=(0, (3, 2)), zorder=1)
    dot(ax, [40.0], [2.0], GOLD, "o", s=22, z=5)
    an = VG["ancestors"]
    xs_us = [100.0 * (1.0 - an[k]["men"]["q"]) for k in ("1900-1939", "1940-1959", "1960-1979")]
    ys_us = [an[k]["ratio"] for k in ("1900-1939", "1940-1959", "1960-1979")]
    dot(ax, xs_us, ys_us, WOMEN, "D", s=20, z=5, label="U.S. cohorts 1900\u201379")
    ax.set_yscale("log")
    ax.set_ylim(0.72, 8.2)
    ax.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlim(18, 100)
    panel(ax, "D", "$N_{e,f}/N_{e,m}=2$ is not half the men (Prop. 18)", xlabel="Men leaving descendants (%)", ylabel="Effective ratio $N_{e,f}/N_{e,m}$")
    ax.legend(loc="upper right", fontsize=5.3, labelspacing=0.17)

    return pdf(fig)


# ---------------------------------------------------------------------------
# 5B. HOLOFACE-L1 43-WAVE BENCHMARK, SHAPLEY & VOTER CEILING (Fig. 9)
# ---------------------------------------------------------------------------
def holoface_ceiling_plot(data):
    hb = data["holoface_benchmark"]
    hp = hb["holoface_pure_zero_metadata"]
    hf = hb["holoface_full_multimodal"]
    pc = hb["psychometric_voter_ceiling"]
    waves = hb["waves_trajectory"]
    mhc = data["open_synthesis"]["published_benchmarks"]["havlicek_2020_mhc"]

    fig, axes = plt.subplots(2, 2, figsize=(7.24, 4.30))
    fig.subplots_adjust(left=0.095, right=0.988, bottom=0.11, top=0.90, wspace=0.38, hspace=0.48)

    # Panel A: 43-Wave Trajectory + HoloFace-Pure & HoloFace-Full Optimum
    ax = axes[0, 0]
    xs = np.arange(1, len(waves) + 1)
    sp_w = [w["test_spearman_r"] for w in waves]
    sp2_w = [r ** 2 for r in sp_w]
    ax.plot(xs, sp_w, color=MEN, lw=1.35, alpha=0.85, label="Wave test Spearman $\\rho$")
    ax.plot(xs, sp2_w, color=VIOLET, lw=1.15, ls="--", alpha=0.80, label="Explained rank var. $\\rho^2$")
    # Highlight H1, HoloFace-Pure, and HoloFace-Full
    dot(ax, [1], [sp_w[0]], GREY, "o", s=22, z=5)
    ax.axhline(hp["test"]["spearman_r"], color=AQUA, lw=1.0, ls=":", zorder=3)
    dot(ax, [len(waves) + 1.5], [hp["test"]["spearman_r"]], AQUA, "s", s=26, z=6, label=f"HoloFace-Pure ($\\rho={hp['test']['spearman_r']:.3f}$)")
    dot(ax, [len(waves) + 1.5], [hf["test_l1"]["spearman_r"]], WOMEN, "D", s=28, z=6, label=f"HoloFace-Full ($\\rho={hf['test_l1']['spearman_r']:.3f}$)")
    ax.set_xlim(0, len(waves) + 3.5)
    ax.set_ylim(0.0, 0.82)
    ax.set_xticks([1, 10, 20, 30, len(waves)], ["H1", "H10", "H20", "H30", f"H{len(waves)}"])
    panel(ax, "A", f"{len(waves)}-wave progression to HoloFace-L1", xlabel="Benchmark wave (10k profiles, 20k portraits)", ylabel="Held-out test $\\rho$ / $\\rho^2$")
    ax.legend(loc="lower right", fontsize=5.5, labelspacing=0.18)
    ax.text(2.0, 0.73, f"HoloFace-Full MAE {hf['test_l1']['beauty_mae']:.4f}\nSingle-img MAE {hf['single_image_test']['beauty_mae']:.4f}", fontsize=5.6, color=INK, fontweight="bold", linespacing=1.1)

    # Panel B: Exact 2^4=16 Shapley Decomposition of Zero-Metadata HoloFace-Pure
    ax = axes[0, 1]
    sh_rows = sorted(hp["shapley_4family_decomposition"], key=lambda r: r["shapley_share_of_pure_face_pct"])
    sh_labels = {
        "F3_staging_pose": "Pose, smile &\nframing (F3)",
        "F2_skin_photometry": "Skin chromophores &\nspectral relief (F2)",
        "F1_geometry_3d": "3D Kendall landmark\nmorphology (F1)",
        "F4_foundation_12space": "12-space foundation &\ncortical vision (F4)",
    }
    sh_cols = {
        "F3_staging_pose": GOLD,
        "F2_skin_photometry": WOMEN,
        "F1_geometry_3d": AQUA,
        "F4_foundation_12space": MEN,
    }
    yp = np.arange(len(sh_rows))
    for i, r in enumerate(sh_rows):
        pct = r["shapley_share_of_pure_face_pct"]
        fid = r["family_id"]
        ax.barh(i, pct, height=0.56, color=sh_cols[fid], alpha=0.90, zorder=3)
        ax.text(pct + 1.2, i, f"{pct:.1f}%", va="center", ha="left", fontsize=6.2, fontweight="bold", color=INK)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(yp, [sh_labels[r["family_id"]] for r in sh_rows], fontsize=6.0)
    ax.set_xlim(0, 76)
    panel(ax, "B", "Exact 4-family Shapley share (zero metadata)", xlabel="Share of HoloFace-Pure rank variance (%)")

    # Panel C: Finite-Voter Psychometric Oracle Ceiling vs HoloFace Across Vote Tiers
    ax = axes[1, 0]
    tkeys = ["votes_20_49", "votes_50_199", "votes_200_plus", "votes_500_plus"]
    tlabs = ["20\u201349\nvotes", "50\u2013199\nvotes", "200+\nvotes", "500+\nvotes"]
    tx = np.arange(len(tkeys))
    or_sp = [pc["tiers"][k]["oracle_finite_vote_spearman_r"] for k in tkeys]
    fu_sp = [pc["tiers"][k]["holoface_full_spearman_r"] for k in tkeys]
    pu_sp = [pc["tiers"][k]["holoface_pure_spearman_r"] for k in tkeys]
    ax.fill_between(tx, fu_sp, or_sp, color=WOMEN, alpha=0.14, lw=0, label="Single-context gap")
    ax.plot(tx, or_sp, color=INK, lw=1.65, marker="D", ms=4.0, label="Finite-voter Oracle")
    ax.plot(tx, fu_sp, color=MEN, lw=1.6, marker="o", ms=3.8, label="HoloFace-Full")
    ax.plot(tx, pu_sp, color=AQUA, lw=1.4, ls="--", marker="s", ms=3.5, label="HoloFace-Pure")
    ax.set_xticks(tx, tlabs, fontsize=6.2)
    ax.set_ylim(0.35, 1.04)
    panel(ax, "C", "Finite-voter Oracle vs. single-context gap", xlabel="Held-out test voter-depth tier", ylabel="Test Spearman $\\rho$")
    ax.legend(loc="lower right", fontsize=5.5, labelspacing=0.18)
    gap500 = 100.0 * pc["tiers"]["votes_500_plus"]["irreducible_single_context_gap_full"]
    ax.text(2.05, 0.855, f"500+ gap:\n{gap500:.1f}% rank var.", fontsize=5.6, color=WOMEN, fontweight="bold", ha="center", linespacing=1.06)

    # Panel D: MHC Chemosensory Null + Skew-Unbiased L2 Male Bias Calibration
    ax = axes[1, 1]
    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (2, 2)))
    labels_mhc = [
        f"Genomic pairing\n(k={mhc[0]['k_effects']})",
        f"Odor preference\n(k={mhc[1]['k_effects']})",
        f"Sexual satisfaction\n(k={mhc[2]['k_effects']})",
        f"Combined synthesis\n(k={mhc[3]['k_effects']})",
    ]
    for i, row in enumerate(mhc):
        est = row["fisher_z"]
        lo, hi = row["ci_95"]
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], color=INK, fmt="D", mfc=GOLD if i < 3 else VIOLET, mec=INK, ms=4.8, elinewidth=1.2, capsize=2.8)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(range(len(mhc)), labels_mhc, fontsize=6.0)
    ax.set_xlim(-0.22, 0.22)
    ax.set_ylim(-0.6, 3.8)
    panel(ax, "D", "MHC null (k=63) & L2 bias calibration", xlabel="Random-effects Fisher $z$ (Havlicek et al. 2020)")
    m_b_l2 = hf["subgroups_test_l2_gender"]["Male"]["bias"]
    rmse_l2 = hf["test_l2_unbiased_head"]["beauty_rmse"]
    ax.text(
        0.21, 3.55,
        f"L2 head: RMSE {rmse_l2:.4f}\nMale bias +0.088\u2192{m_b_l2:+.3f}",
        fontsize=5.4, color=INK, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID, lw=0.5, alpha=0.92),
    )

    return pdf(fig)


# ---------------------------------------------------------------------------
# 6B. COMPLETED-FERTILITY FLOOR, WORDSUM, IDEALS & HCMST GATE (Fig. 15)
# ---------------------------------------------------------------------------
def gss_fertility_gate_plot(data):
    G = _v9_json("gss")
    V = _v9_json("e19_village")
    W = _v9_json("e20_world")
    osyn = data["open_synthesis"]
    gss_all = osyn["v9_anchors"]["gss_three_episodes_all"]
    hcm = osyn["hcmst_2017"]

    fig, axes = plt.subplots(2, 3, figsize=(7.24, 4.86))
    fig.subplots_adjust(left=0.088, right=0.985, bottom=0.105, top=0.905, wspace=0.44, hspace=0.52)

    def _plot_mw(ax, xs, men_s, wom_s, m_lbl="Men", w_lbl="Women"):
        for ser, col, mk, off, lb in ((wom_s, WOMEN, "D", 0.05, w_lbl), (men_s, MEN, "o", -0.05, m_lbl)):
            x = np.asarray(xs, dtype=float) + off
            pt = [v[0] for v in ser]
            lo = [v[1] for v in ser]
            hi = [v[2] for v in ser]
            ax.plot(x, pt, color=col, lw=1.45, zorder=2)
            ax.vlines(x, lo, hi, color=col, lw=0.95, zorder=2)
            dot(ax, x, pt, col, mk, s=15 if mk == "o" else 13, z=3, label=lb)

    # Panel A: Completed Childlessness by Personal Income Fifth (Ages 45-79)
    ax = axes[0, 0]
    fq = G["fifths"]["income"]
    pc3 = lambda v: [100.0 * v[0], 100.0 * v[1], 100.0 * v[2]]
    _plot_mw(ax, range(5), [pc3(v) for v in fq["men"]["childless"]], [pc3(v) for v in fq["women"]["childless"]])
    ax.set_xticks(range(5), ["Bot.\n20%", "Q2", "Q3", "Q4", "Top\n20%"], fontsize=5.8)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(0, 27)
    panel(ax, "A", "Childlessness by income fifth", xlabel="Personal income fifth (ages 45\u201379)", ylabel="Childless (%)", fontsize=7.5)
    ax.legend(loc="upper right", fontsize=5.5)

    # Panel B: Pairing (+) vs. Within-Marriage Births (-) Across Income, Prestige, Schooling & Objective WORDSUM
    ax = axes[0, 1]
    ax.axvline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    ep_traits = [
        ("income", "Personal\nincome"),
        ("prestige", "Occup.\nprestige"),
        ("schooling", "Years of\nschooling"),
        ("vocabulary", "WORDSUM\n(g-proxy)"),
    ]
    yb = np.arange(len(ep_traits))
    for i, (tk, _) in enumerate(ep_traits):
        mp = gss_all[tk]["men"]["pairing"]
        mn = gss_all[tk]["men"]["number"]
        wn = gss_all[tk]["women"]["number"]
        ax.errorbar(
            mp[0], i - 0.17, xerr=[[mp[0] - mp[1]], [mp[2] - mp[0]]],
            color=AQUA, fmt="o", ms=3.8, elinewidth=1.0, capsize=2.0, label="Men: pairing (+)" if i == 0 else None,
        )
        ax.errorbar(
            mn[0], i + 0.02, xerr=[[mn[0] - mn[1]], [mn[2] - mn[0]]],
            color=MEN, fmt="s", mfc="white", ms=3.4, elinewidth=0.95, capsize=1.8, label="Men: marital births (\u2212)" if i == 0 else None,
        )
        ax.errorbar(
            wn[0], i + 0.21, xerr=[[wn[0] - wn[1]], [wn[2] - wn[0]]],
            color=WOMEN, fmt="D", mfc="white", ms=3.3, elinewidth=0.95, capsize=1.8, label="Women: marital births (\u2212)" if i == 0 else None,
        )
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(yb, [x[1] for x in ep_traits], fontsize=5.6)
    ax.set_ylim(3.48, -1.28)
    ax.set_xlim(-0.16, 0.08)
    panel(ax, "B", "Status & WORDSUM: +pair, \u2212births", xlabel="Standardized episode gradient (\u03b2)", fontsize=7.4)
    ax.legend(loc="upper left", fontsize=4.5, labelspacing=0.12, handletextpad=0.18, frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")

    # Panel C: Childlessness Share of Opportunity I + Non-Marital Parenthood Gate
    ax = axes[0, 2]
    cohs = ["1900-1919", "1920-1939", "1940-1959", "1960-1979"]
    opp = G["opportunity"]
    gate = G["gate"]
    _plot_mw(
        ax, range(4),
        [pc3(opp["men"][c]["share_childless"]) for c in cohs],
        [pc3(opp["women"][c]["share_childless"]) for c in cohs],
        m_lbl="M childless / $I$", w_lbl="W childless / $I$",
    )
    ax.plot(range(4), [100.0 * gate["women"][c]["parents_nm"][0] for c in cohs], color=WOMEN, lw=1.15, ls="--", marker="D", mfc="white", ms=2.9, label="W parent | unmarr.")
    ax.plot(range(4), [100.0 * gate["men"][c]["parents_nm"][0] for c in cohs], color=MEN, lw=1.15, ls="--", marker="o", mfc="white", ms=2.9, label="M parent | unmarr.")
    ax.set_xticks(range(4), ["'00\u201319", "'20\u201339", "'40\u201359", "'60\u201379"], fontsize=5.8)
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(0, 68)
    panel(ax, "C", "Selection floor & marital gate", xlabel="U.S. birth cohort", ylabel="Share (%)", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=4.65, labelspacing=0.14)

    # Panel D: Unchanged Family-Size Ideals vs Realized Birth Shortfall (1972-2024)
    ax = axes[1, 0]
    pers = ["1972-1984", "1985-1999", "2000-2011", "2012-2018", "2021-2024"]
    xl = ["'72\u201384", "'85\u201399", "'00\u201311", "'12\u201318", "'21\u201324"]
    id_d = V["ideals"]
    for key, ci_k, col, mk, lb in (("ideal", "ideal_ci", AQUA, "o", "Stated ideal (18\u201344)"), ("actual", "actual_ci", INK, "s", "Completed births (40\u201349)")):
        yv = [id_d[p][key] for p in pers]
        ax.plot(range(5), yv, color=col, lw=1.5, zorder=2)
        ax.vlines(range(5), [id_d[p][ci_k][0] for p in pers], [id_d[p][ci_k][1] for p in pers], color=col, lw=0.95, zorder=2)
        dot(ax, range(5), yv, col, mk, s=14, label=lb)
    ax.set_xticks(range(5), xl, fontsize=5.6)
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(1.4, 3.48)
    panel(ax, "D", "Ideals ~2.7; non-marriage = 94%", xlabel="GSS survey period", ylabel="Children per adult", fontsize=7.5)
    ax.legend(loc="lower left", fontsize=4.9)
    bl = V["births"]["long"]["all"]
    m_sh = 100.0 * bl["marriage"] / bl["change"]
    ax.text(4.25, 3.34, f"Ages 25\u201339: {bl['change']:+.2f}\n({m_sh:.0f}% non-marriage)", fontsize=4.9, color=VIOLET, fontweight="bold", ha="right", va="top", linespacing=1.04)

    # Panel E: Witness Ties & Screen Hours vs Partnered Probability (Ages 22-34)
    ax = axes[1, 1]
    assoc = V["village"]["assoc"]
    scr = W["screens"]
    rows_d = [
        ("Relatives monthly", assoc["relatives_monthly"]["points"], assoc["relatives_monthly"]["ci"], AQUA),
        ("Worship weekly", assoc["worship_weekly"]["points"], assoc["worship_weekly"]["ci"], VIOLET),
        ("Neighbors monthly", assoc["neighbours_monthly"]["points"], assoc["neighbours_monthly"]["ci"], MEN),
        ("Friends monthly", assoc["friends_monthly"]["points"], assoc["friends_monthly"]["ci"], GOLD),
        ("Social media/hr (M)", -scr["men"]["points_per_hour"], [-scr["men"]["ci"][1], -scr["men"]["ci"][0]], GREY),
        ("Social media/hr (W)", -scr["women"]["points_per_hour"], [-scr["women"]["ci"][1], -scr["women"]["ci"][0]], WOMEN),
    ]
    ax.axvline(0, color=INK2, lw=0.75, ls=(0, (2, 2)))
    yd = [i + (0.35 if i >= 4 else 0.0) for i in range(len(rows_d))]
    for yi, (_, v, ci, col) in zip(yd, rows_d):
        ax.plot(ci, [yi, yi], color=col, lw=1.25, zorder=2)
        dot(ax, [v], [yi], col, "o", s=16, z=3)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(yd, [r[0] for r in rows_d], fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlim(-12.5, 15.0)
    panel(ax, "E", "Kin witnesses vs. screen hours", xlabel="Partnered diff. (pp, ages 22\u201334)", fontsize=7.5)

    # Panel F: Stanford HCMST 70-Year Disintermediation of Courtship (N=3,295 Meeting-Coded Couples)
    ax = axes[1, 2]
    era_keys = ["pre_1980", "1980_1994", "1995_2004", "2005_2011", "2012_2017"]
    era_lbls = ["Pre-'80", "'80\u201394", "'95\u201304", "'05\u201311", "'12\u201317"]
    ex = np.arange(len(era_keys))
    eras = hcm["historical_eras"]
    yr_pre = [eras[k]["mean_pre_romance_years"] for k in era_keys]
    on_sh = [100.0 * eras[k]["online_all_share"] for k in era_keys]
    ap_sh = [100.0 * eras[k]["online_dating_app_share"] for k in era_keys]
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(MUTED)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.tick_params(labelsize=5.8, colors=INK2)
    ax.plot(ex, yr_pre, color=AQUA, lw=1.6, marker="D", ms=3.3, label="Pre-romance yrs (left)")
    ax2.plot(ex, on_sh, color=GOLD, lw=1.45, ls="--", marker="o", ms=3.0, label="Met online % (right)")
    ax2.plot(ex, ap_sh, color=WOMEN, lw=1.2, ls=":", marker="^", mfc="white", ms=2.9)
    ax.set_xticks(ex, era_lbls, fontsize=5.5)
    ax.set_ylim(0.0, 3.35)
    ax2.set_ylim(0, 44)
    ax2.set_ylabel("Met online / apps (%)", fontsize=5.8, color=INK2)
    panel(ax, "F", "70-yr audit collapse (HCMST)", xlabel="Year couple met", ylabel="Pre-romance yrs", fontsize=7.5)
    ax.text(
        1.05, 3.12,
        f"2.90\u21920.42 yr acquaintance;\n'12\u201317: {on_sh[-1]:.0f}% online ({ap_sh[-1]:.0f}% apps)",
        fontsize=4.65, color=INK, va="top", ha="left", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )

    return pdf(fig)


# ---------------------------------------------------------------------------
# 3. MALE HEIGHT PERCENTILE VS ATTRACTION, 6'0" STAGING & FITNESS (NEW SHOWSTOPPER)
# ---------------------------------------------------------------------------
def height_attraction_plot(data):
    whyte_h = data["open_synthesis"]["whyte_2021"]["height_vs_attraction"]
    pov_h = data["pov"]["height_percentiles"]["by_sex"]["Male"]
    lid = data["open_synthesis"]["lidborg_2022"]
    okc = data["open_synthesis"]["okcupid_jse"]
    ost = okc["orientation_staging_experiment"]
    tpp = okc["ten_prompt_portfolio"]

    fig, axes = plt.subplots(2, 3, figsize=(7.24, 4.78))
    fig.subplots_adjust(left=0.072, right=0.988, bottom=0.105, top=0.905, wspace=0.45, hspace=0.49)

    # Panel A: Height Percentile vs Mean Attraction Percentile (Whyte 2021, N=7,060)
    ax = axes[0, 0]
    ax.axhline(50, color=MUTED, lw=0.65, ls=(0, (2, 2)))
    for key, label, col, mk in (("Men", "Men ($n=4{,}375$)", MEN, "o"), ("Women", "Women ($n=2{,}685$)", WOMEN, "D")):
        br = [b for b in whyte_h[key]["by_bracket"] if b["n"] >= 20]
        xs = [b["height_percentile"] for b in br]
        ys = [b["mean_attraction_percentile"] for b in br]
        lo = [b["attraction_percentile_ci_95"][0] for b in br]
        hi = [b["attraction_percentile_ci_95"][1] for b in br]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.14, lw=0)
        ax.plot(xs, ys, color=col, marker=mk, ms=3.6, lw=1.55, label=label)
    mv = whyte_h["Men"]["multivariate_standardized"]["attraction_pct_per_sd"]
    qfit = whyte_h["Men"]["quadratic_fit_pct_per_sd"]["quadratic"]
    ax.set_xlim(-2, 102)
    ax.set_ylim(28, 62)
    panel(ax, "A", "Height vs attraction pct (Whyte 2021)", xlabel="Within-sex height percentile", ylabel="Mean attraction percentile", fontsize=7.4)
    ax.legend(loc="upper left", fontsize=5.5)
    ax.text(
        3.0, 29.6,
        f"Curvilinearity $\\beta_2 = {qfit['estimate']:.2f}$ [{qfit['ci_95'][0]:.2f}, {qfit['ci_95'][1]:.2f}]\n"
        f"Health +{mv['health']['estimate']:.1f} vs height +{mv['height']['estimate']:.2f} pct/SD",
        fontsize=5.0, color=INK2, linespacing=1.1,
    )

    # Panel B: The 6'0" Staged-Signal Cliff Across 781,680 POV Men
    ax = axes[0, 1]
    rows_in = [r for r in pov_h["by_inch"] if 62 <= r["inches"] <= 78]
    inches = np.array([r["inches"] for r in rows_in])
    obs_pct = np.array([100 * r["observed_share"] for r in rows_in])
    exp_pct = np.array([100 * r["gaussian_expected_share"] for r in rows_in])
    bar_colors = [GOLD if inc == 72 else (RED if inc == 73 else MEN) for inc in inches]
    ax.bar(inches, obs_pct, width=0.72, color=bar_colors, alpha=0.84, label="Observed profiles")
    ax.plot(inches, exp_pct, color=INK, lw=1.45, ls="--", marker="o", ms=2.3, label="Gaussian expected")
    tick_in = [64, 67, 70, 72, 75]
    tick_lb = ["5'4\"", "5'7\"", "5'10\"", "6'0\"", "6'3\""]
    ax.set_xticks(tick_in, tick_lb)
    ax.set_ylim(0, 15.8)
    cliff = pov_h["cliff_5ft11_6ft0_6ft1"]
    panel(ax, "B", "6'0\" staged cliff (781,680 POV men)", xlabel="Self-reported male height", ylabel="Share of male profiles (%)", fontsize=7.4)
    ax.legend(loc="upper left", fontsize=5.3)
    ax.annotate(
        f"6'0\": {cliff['n_6ft0']:,} (1.20\u00d7)\n6'1\": {cliff['n_6ft1']:,} ({cliff['drop_6ft0_to_6ft1_pct']:.1f}%)",
        xy=(72.4, 11.2), xytext=(72.8, 13.3),
        fontsize=5.1, color=INK, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.65),
    )

    # Panel C: Audience-Targeted Staging Natural Experiment (OkCupid-JSE Straight vs. Gay Men)
    ax = axes[0, 2]
    inc_grid = list(range(67, 76))
    str_st = ost["groups"]["men_straight"]["stature"]
    gay_st = ost["groups"]["men_gay"]["stature"]
    str_sh = [100.0 * str_st["by_inch_67_to_75"][str(i)]["share"] for i in inc_grid]
    gay_sh = [100.0 * gay_st["by_inch_67_to_75"][str(i)]["share"] for i in inc_grid]
    ax.axvline(72, color=GOLD, lw=0.95, ls=(0, (2, 2)), zorder=1)
    ax.plot(inc_grid, str_sh, color=MEN, lw=1.7, marker="o", ms=3.6, label=f"Straight men ($n={ost['groups']['men_straight']['n']:,}$)")
    ax.plot(inc_grid, gay_sh, color=AQUA, lw=1.55, ls="--", marker="D", ms=3.3, label=f"Gay men ($n={ost['groups']['men_gay']['n']:,}$)")
    ax.scatter([72], [str_sh[inc_grid.index(72)]], color=GOLD, edgecolor=INK, s=34, zorder=5)
    ax.set_xticks([67, 69, 71, 72, 73, 75], ["5'7\"", "5'9\"", "5'11\"", "6'0\"", "6'1\"", "6'3\""])
    ax.set_xlim(66.6, 75.4)
    ax.set_ylim(1.4, 19.2)
    panel(ax, "C", "Audience staging: straight vs gay men", xlabel="OkCupid SF reported height", ylabel="Share of valid male profiles (%)", fontsize=7.4)
    ax.legend(loc="upper left", fontsize=5.0, labelspacing=0.16)
    inc_sm = 100.0 * ost["groups"]["men_straight"]["income_disclosure"]["disclosed_positive_share"]
    inc_gm = 100.0 * ost["groups"]["men_gay"]["income_disclosure"]["disclosed_positive_share"]
    inc_sw = 100.0 * ost["groups"]["women_straight"]["income_disclosure"]["disclosed_positive_share"]
    ax.text(
        66.85, 2.15,
        f"6'0\" surge: +{str_st['jump_5ft11_to_6ft0_pct']:.1f}% Str vs \u2212{abs(gay_st['jump_5ft11_to_6ft0_pct']):.1f}% Gay\nIncome shown: {inc_sm:.1f}% Str vs {inc_gm:.1f}% Gay",
        fontsize=4.65, color=INK, ha="left", va="bottom", linespacing=1.06,
        bbox=dict(boxstyle="round,pad=0.14", fc=BAND, ec="none", alpha=0.93),
    )

    # Panel D: POV Male Height Percentile vs Union Durability, Casual Strategy & Parenthood
    ax = axes[1, 0]
    valid_rows = [r for r in pov_h["by_inch"] if r["outcomes_ages_30_49"] and r["outcomes_ages_30_49"]["n"] >= 300]
    hpct = [r["mean_height_percentile"] for r in valid_rows]
    u3y = [100 * r["outcomes_ages_30_49"]["over_3y_union"]["share"] for r in valid_rows]
    par = [100 * r["outcomes_ages_30_49"]["parenthood"]["share"] for r in valid_rows]
    cas = [100 * r["outcomes_ages_30_49"]["casual_intent"]["share"] for r in valid_rows]
    ax.plot(hpct, u3y, color=VIOLET, lw=1.6, marker="o", ms=2.9, label="Sustained >3-yr union")
    ax.plot(hpct, par, color=MEN, lw=1.45, marker="s", ms=2.7, label="Has children")
    ax.plot(hpct, cas, color=GOLD, lw=1.45, marker="^", ms=2.9, label="Casual-only intent")
    ax.set_xlim(-2, 102)
    ax.set_ylim(10, 88)
    panel(ax, "D", "POV outcomes by height pct (30\u201349)", xlabel="Male height percentile", ylabel="Share of male profiles (%)", fontsize=7.4)
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.76), fontsize=5.3, labelspacing=0.18)

    # Panel E: Lidborg, Cross & Boothroyd (2022) Mating vs Reproduction Meta-Analysis
    ax = axes[1, 1]
    ax.axvline(0, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    groups = [
        ("Height: Fertility\n(k=35, N=23.0k)", lid["height"]["reproduction_all"], [e["r_oriented"] for e in lid["height"]["effects"] if e["grouping"] == "REP"], MEN, "o"),
        ("Height: Mating\n(k=62, N=92.5k)", lid["height"]["mating"], [e["r_oriented"] for e in lid["height"]["effects"] if e["grouping"] == "MAT"], B300, "o"),
        ("Strength: Fertility\n(k=14, N=1.4k)", lid["body_strength"]["reproduction_all"], [e["r_oriented"] for e in lid["body_strength"]["effects"] if e["grouping"] == "REP"], AQUA, "D"),
        ("Strength: Mating\n(k=121, N=38.3k)", lid["body_strength"]["mating"], [e["r_oriented"] for e in lid["body_strength"]["effects"] if e["grouping"] == "MAT"], VIOLET, "D"),
    ]
    rng_jit = np.random.default_rng(20261111)
    for idx_g, (glab, summary, pts, col, mk) in enumerate(groups):
        jit = idx_g + rng_jit.uniform(-0.18, 0.18, size=len(pts))
        ax.scatter(pts, jit, s=8, color=col, alpha=0.25, edgecolors="none", zorder=2)
        est = summary["re_mean_r"]
        lo, hi = summary["re_ci_95"]
        ax.errorbar(est, idx_g, xerr=[[est - lo], [hi - est]], color=INK, fmt=mk, mfc=col, mec=INK, ms=4.8, elinewidth=1.25, capsize=2.6, zorder=5)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=5.3)
    ax.set_xlim(-0.42, 0.55)
    panel(ax, "E", "Mating vs fertility (Lidborg 2022)", xlabel="Fitness-oriented $r$ (95% RE CI)", fontsize=7.4)

    # Panel F: OkCupid 47,293 "Message me if..." & 48,413 "First noticed" Cues
    ax = axes[1, 2]
    e9 = tpp["essay9_message_me_if"]
    e3 = tpp["essay3_first_noticed"]["themes"]
    bars_f = [
        ("Rich / 6-fig", 100.0 * e9["themes"]["rich_wealthy_sixfig"]["share"], RED),
        ("Tall / 6'0\"+", 100.0 * e9["themes"]["tall_6ft_height"]["share"], GOLD),
        ("Adventure", 100.0 * e9["themes"]["adventure_outdoors"]["share"], B300),
        ("Mind / wit", 100.0 * e9["themes"]["curiosity_intelligence"]["share"], MEN),
        ("Kind / sincere", 100.0 * e9["themes"]["kindness_sincerity"]["share"], VIOLET),
        ("Humor / laugh", 100.0 * e9["themes"]["humor_banter"]["share"], AQUA),
    ]
    yf = np.arange(len(bars_f))
    ax.barh(yf, [b[1] for b in bars_f], height=0.62, color=[b[2] for b in bars_f], alpha=0.88)
    for yi, (_, val_f, _) in enumerate(bars_f):
        ax.text(val_f + 0.35, yi, f"{val_f:.1f}%" if val_f >= 2.0 else f"{val_f:.2f}%", va="center", fontsize=5.0, color=INK)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_yticks(yf, [b[0] for b in bars_f], fontsize=5.4)
    ax.set_xlim(0, 22.8)
    panel(ax, "F", "47.3k 'Message me if' vs noticed cues", xlabel="Share of 'Message me if' essays (%)", fontsize=7.4)
    sm_p = 100.0 * e3["smile_laugh"]["share"]
    ey_p = 100.0 * e3["eyes"]["share"]
    ht_p = 100.0 * e3["height"]["share"]
    ax.text(
        22.3, 0.20,
        f"Humor: {e9['ratios']['humor_to_tall']:.1f}\u00d7 tall, {e9['ratios']['humor_to_wealth']:.1f}\u00d7 wealth;\nFirst noticed: smile {sm_p:.0f}% vs height {ht_p:.0f}%",
        fontsize=4.65, color=INK, ha="right", va="bottom", linespacing=1.06,
        bbox=dict(boxstyle="round,pad=0.14", fc=BAND, ec="none", alpha=0.93),
    )

    return pdf(fig)


# ---------------------------------------------------------------------------
# 4. GLOBAL IDEALS, TRUST, CULTURE & PLATFORM BRIDGE (performance-fig-world-synthesis.pdf)
# ---------------------------------------------------------------------------
def world_synthesis_plot(data):
    osyn = data["open_synthesis"]
    whyte_s = osyn["whyte_2021"]["stated_importance"]
    kow = osyn["kowal_2025"]
    cn = data["hon"]["cross_national"]

    fig, axes = plt.subplots(2, 3, figsize=(7.24, 4.90))
    fig.subplots_adjust(left=0.078, right=0.99, bottom=0.10, top=0.90, wspace=0.40, hspace=0.48)

    # Panel A: Whyte 2021 9-Trait Stated Importance Dumbbell Hierarchy (N = 7,060)
    ax = axes[0, 0]
    w_pool = whyte_s["pooled"]["Women"]
    m_pool = whyte_s["pooled"]["Men"]
    trait_labels = [
        ("trust", "Trust"),
        ("emotional_connection", "Emotional conn."),
        ("openness", "Openness"),
        ("intelligence", "Intelligence"),
        ("attractiveness", "Attractiveness"),
        ("physical_build", "Physical build"),
        ("education", "Education"),
        ("age", "Age"),
        ("income", "Income"),
    ]
    trait_labels.sort(key=lambda kv: 0.5 * (w_pool[kv[0]]["mean"] + m_pool[kv[0]]["mean"]))
    yp_a = np.arange(len(trait_labels))
    for i, (tk, tlab) in enumerate(trait_labels):
        wm = w_pool[tk]["mean"]
        mm = m_pool[tk]["mean"]
        ax.plot([mm, wm], [i, i], color=INK2, lw=1.0, alpha=0.65, zorder=2)
        dot(ax, [mm], [i], MEN, "o", s=16, z=3, label="Men" if i == 0 else None)
        dot(ax, [wm], [i], WOMEN, "D", s=15, z=4, label="Women" if i == 0 else None)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, lw=0.55)
    ax.set_yticks(yp_a, [t[1] for t in trait_labels], fontsize=5.8)
    ax.set_xlim(18, 92)
    panel(ax, "A", "Trust & connection lead (7.1k)", xlabel="Stated importance (0–100)")
    ax.legend(loc="lower right", fontsize=5.8, handletextpad=0.25)

    # Panel B: Kowal 53-Country Compositional Budget Sex Differences
    ax = axes[0, 1]
    items = [("resources", "Financial"), ("kindness", "Kindness"), ("age", "Age"), ("religion", "Religion"), ("health", "Health"), ("looks", "Looks")]
    ax.axvline(0, color=INK2, lw=0.75, ls=(0, (2, 2)))
    rng_j = np.random.default_rng(7)
    c_means = kow["cross_country_53"]["country_means"]
    for idx_i, (ikey, ilab) in enumerate(items):
        diffs = [v["women_minus_men"][ikey] for v in c_means.values()]
        col = WOMEN if np.median(diffs) > 0 else MEN
        ax.scatter(diffs, idx_i + rng_j.uniform(-0.18, 0.18, len(diffs)), s=10, color=col, alpha=0.38, edgecolors="none")
        ax.plot([np.median(diffs)] * 2, [idx_i - 0.28, idx_i + 0.28], color=INK, lw=1.8)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_yticks(range(len(items)), [x[1] for x in items])
    ax.set_xlim(-2.4, 2.6)
    panel(ax, "B", "53-country budget tilts (Kowal 2025)", xlabel="Women minus men (points of 30)")

    # Panel C: Cross-Country Male-Female Cultural Co-Variation
    ax = axes[0, 2]
    ax.plot([0.8, 9.2], [0.8, 9.2], color=MUTED, lw=0.7, ls=(0, (2, 2)))
    for ikey, col, mk, lab in (("religion", VIOLET, "o", "Religion ($r=.94$)"),
                               ("kindness", AQUA, "D", "Kindness ($r=.94$)"),
                               ("looks", GOLD, "^", "Looks ($r=.75$)")):
        mx = [v["men"][ikey] for v in c_means.values()]
        wy = [v["women"][ikey] for v in c_means.values()]
        dot(ax, mx, wy, col, mk, s=13, alpha=0.82, label=lab)
    ax.set_xlim(0.8, 9.2)
    ax.set_ylim(0.8, 9.2)
    panel(ax, "C", "Culture shifts both sexes together", xlabel="Men's country mean", ylabel="Women's country mean")
    ax.legend(loc="lower right", fontsize=5.7)

    # Panel D: Partnering ("Door") Slopes Across 90,516 Adults
    ax = axes[1, 0]
    dtraits = [("resources", "Resources"), ("looks", "Looks"), ("health", "Health"), ("kindness", "Kindness")]
    ypos = np.arange(len(dtraits))
    ax.axvline(0, color=MUTED, lw=0.7)
    for (s_code, s_lab, col, mk), off in zip((("women", "Women", WOMEN, "D"), ("men", "Men", MEN, "o")), (-0.12, 0.12)):
        ests = [kow["partnering_slopes"]["by_sex"][s_code][k]["pp_per_sd"] for k, _ in dtraits]
        los = [kow["partnering_slopes"]["by_sex"][s_code][k]["ci_95"][0] for k, _ in dtraits]
        his = [kow["partnering_slopes"]["by_sex"][s_code][k]["ci_95"][1] for k, _ in dtraits]
        ax.errorbar(ests, ypos + off, xerr=[[e - l for e, l in zip(ests, los)], [h - e for e, h in zip(ests, his)]],
                    color=col, fmt=mk, ms=3.9, elinewidth=0.95, capsize=2, label=s_lab)
    ax.grid(False)
    ax.grid(axis="x", color=GRID, linewidth=0.55)
    ax.set_yticks(ypos, [x[1] for x in dtraits])
    ax.set_xlim(-2.5, 9.5)
    panel(ax, "D", "Having a partner per self-rated SD", xlabel="Change in partnered share (pp)")
    ax.legend(loc="lower right", fontsize=6.1)

    # Panel E: HON Cross-National Age-Beauty Drop (82 Strict Countries/Territories)
    ax = axes[1, 1]
    strict_rows = [v for v in cn["countries"].values() if v["female_minus_male_age_drop_pp"] is not None]
    mx = [v["by_sex"]["Male"]["age_drop_pp"] for v in strict_rows]
    wy = [v["by_sex"]["Female"]["age_drop_pp"] for v in strict_rows]
    ax.plot([-15, 65], [-15, 65], color=MUTED, lw=0.75, ls=(0, (2, 2)))
    dot(ax, mx, wy, WOMEN, "o", s=14, alpha=0.78)
    ax.set_xlim(-12, 62)
    ax.set_ylim(-12, 62)
    panel(ax, "E", "HON age-rank drop (82 countries)", xlabel="Men's drop 18-29 to 40-59 (pp)", ylabel="Women's drop (pp)")
    st_sum = cn["summary_strict_age_contrast"]
    ax.text(-9, 53, f"Women steeper in\n{st_sum['female_steeper_drop_countries']}/{st_sum['countries']} ({100*st_sum['female_steeper_drop_countries']/st_sum['countries']:.0f}%)",
            fontsize=5.9, color=INK, linespacing=1.12)

    # Panel F: HON Participation Ratio vs Raw Like-Rate Sex Gap (75 Countries >=500/sex)
    ax = axes[1, 2]
    c500 = [(k, v) for k, v in cn["countries"].items() if v["by_sex"]["Female"]["n"] >= 500 and v["by_sex"]["Male"]["n"] >= 500]
    lx = np.array([math.log10(v["female_to_male_profile_ratio"]) for _, v in c500])
    gy = np.array([v["female_minus_male_like_rate_pp"] for _, v in c500])
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    dot(ax, lx, gy, MEN, "o", s=14, alpha=0.75)
    slope, icpt = np.polyfit(lx, gy, 1)
    xx_l = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(xx_l, slope * xx_l + icpt, color=INK, lw=1.25)
    for tag, cname, dx, dy in (("US", "United States", 0.04, -3.2), ("BR", "Brazil", -0.22, 2.2), ("UK", "United Kingdom", 0.04, 1.8)):
        if cname in cn["countries"]:
            vc = cn["countries"][cname]
            ax.text(math.log10(vc["female_to_male_profile_ratio"]) + dx, vc["female_minus_male_like_rate_pp"] + dy, tag, fontsize=5.8, fontweight="bold", color=INK)
    r_part = kow["kowal_hon_bridge"]["hon_500_log_participation_vs_like_gap_r"]
    panel(ax, "F", f"Upload self-selection ($r={r_part:.2f}$)", xlabel="$\\log_{10}$(Female / Male profiles)", ylabel="F minus M like rate (pp)")

    return pdf(fig)


# ---------------------------------------------------------------------------
# 4B. THE ARCHITECTURE OF HUMAN LOVE (performance-fig-love-bond.pdf, Fig. 8)
# ---------------------------------------------------------------------------
def love_bond_plot(data):
    osyn = data["open_synthesis"]
    kow = osyn["kowal_2025"]
    joel = osyn["joel_2020"]
    sccs = osyn["betzig_sccs"]
    hcm = osyn["hcmst_2017"]
    swv = data["pov"]["self_written_voice"]

    fig = plt.figure(figsize=(7.24, 5.04))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.06, 1.08, 1.04], height_ratios=[1.02, 1.0], wspace=0.50, hspace=0.49)
    fig.subplots_adjust(left=0.098, right=0.988, bottom=0.098, top=0.918)

    # Panel A: 47,932-Couple Horse Race Across Cultural Regimes (Kindness vs. Checklist Fit vs. Marrying Up)
    ax_a = fig.add_subplot(outer[0, 0])
    chk_d = kow["checklist_and_surplus_decomposition"]
    infd = kow["pair_bond_reversal"]["reported_infidelity"]["by_sex"]
    preds_a = [
        ("partner_kindness", "Partner kindness"),
        ("partner_looks", "Partner looks"),
        ("idiographic_checklist_fit", "5-trait checklist fit"),
        ("partner_resources", "Partner money"),
        ("marrying_up_surplus", "\"Marrying up\" surplus"),
    ]
    ya = np.arange(len(preds_a))
    ax_a.axvline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    reg_specs = [
        ("pooled", "Pooled (47.9k)", INK, "D", 0.0, True),
        ("high_gender_equality", "High equality", AQUA, "o", -0.18, False),
        ("collectivist", "Collectivist", WOMEN, "s", 0.18, False),
    ]
    for rkey, rlbl, rcol, rmk, roff, is_pool in reg_specs:
        src = chk_d["pooled"] if is_pool else chk_d["by_cultural_regime"][rkey]
        ps = src["predictors_sd_per_sd"]
        ests = [ps[pk]["estimate"] for pk, _ in preds_a]
        los = [ps[pk]["ci_95"][0] for pk, _ in preds_a]
        his = [ps[pk]["ci_95"][1] for pk, _ in preds_a]
        ax_a.errorbar(
            ests, ya + roff, xerr=[[e - l for e, l in zip(ests, los)], [h - e for e, h in zip(ests, his)]],
            color=rcol, fmt=rmk, ms=3.8 if is_pool else 3.0, mfc=rcol if is_pool else "white",
            elinewidth=1.05 if is_pool else 0.85, capsize=1.8, label=rlbl, zorder=5 if is_pool else 3,
        )
    ax_a.grid(False)
    ax_a.grid(axis="x", color=GRID, lw=0.55)
    ax_a.set_yticks(ya, [p[1] for p in preds_a], fontsize=5.3)
    ax_a.invert_yaxis()
    ax_a.set_xlim(-0.07, 0.36)
    k_rat = chk_d["pooled"]["kindness_to_checklist_ratio"]
    inf_kw = infd["women"]["traits"]["kindness"]["pp_per_sd"]
    ax_a.text(
        0.352, 4.35,
        f"Kindness = {k_rat:.1f}\u00d7 checklist;\nW infidelity: {inf_kw:.1f} pp/SD",
        fontsize=4.65, color=INK, ha="right", va="bottom", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_a, "A", "Kindness beats checklist & marry-up", xlabel="Bond quality slope (PRQC SD / SD)", fontsize=7.5)
    ax_a.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), fontsize=4.65, labelspacing=0.14)

    # Panel B: 43 Longitudinal Couples Studies (Joel et al. 2020 PNAS, N=11,196 couples)
    ax_b = fig.add_subplot(outer[0, 1])
    sat_m = joel["satisfaction"]["models"]
    com_m = joel["commitment"]["models"]
    j_rows = [
        ("ActorRelBaseline", "Partner commit.\n& apprec. (base)"),
        ("ActorRelFollowup", "Partner commit.\n(1\u20134y follow-up)"),
        ("ActorTraitBaseline", "Own traits &\nattachment"),
        ("PartnerTraitBaseline", "Partner traits\n(base)"),
        ("PartnerTraitFollowup", "Partner traits\n(1\u20134y follow-up)"),
    ]
    yb = np.arange(len(j_rows))
    for i, (mkey, _) in enumerate(j_rows):
        s_v = sat_m[mkey]["unweighted_mean_r2_pct"]
        c_v = com_m[mkey]["unweighted_mean_r2_pct"]
        col_b = VIOLET if "ActorRel" in mkey else (B300 if "ActorTrait" in mkey else GOLD)
        ax_b.barh(i - 0.14, s_v, height=0.26, color=col_b, alpha=0.92, label="Satisfaction $R^2$" if i == 0 else None)
        ax_b.barh(i + 0.14, c_v, height=0.26, color=AQUA if "ActorRel" in mkey else GREY, alpha=0.82, label="Commitment $R^2$" if i == 0 else None)
        ax_b.text(max(s_v, c_v) + 1.0, i, f"{s_v:.1f}%", va="center", ha="left", fontsize=5.1, fontweight="bold", color=INK)
    ax_b.grid(False)
    ax_b.grid(axis="x", color=GRID, lw=0.55)
    ax_b.set_yticks(yb, [r[1] for r in j_rows], fontsize=5.1)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 58)
    ratio_j = joel["satisfaction"]["actor_rel_to_partner_trait_baseline_ratio"]
    panel(ax_b, "B", f"43 longit. studies: bond {ratio_j:.1f}\u00d7 traits", xlabel="Explained variance $R^2$ (%)", fontsize=7.5)
    ax_b.legend(loc="lower right", fontsize=4.9)

    # Panel C: Stanford HCMST 9.7x App Witness-Audit Compression + Intact Quality Boundary Check
    ax_c = fig.add_subplot(outer[0, 2])
    bv = hcm["by_venue"]
    v_rows = [
        ("school_college", "School / college", AQUA),
        ("family", "Family / kin", VIOLET),
        ("church_worship", "Church / worship", B550),
        ("coworkers", "Coworkers", B300),
        ("friends", "Friends", MEN),
        ("bar_restaurant", "Bar / restaurant", GREY),
        ("online_dating_app", "Dating apps", GOLD),
    ]
    yc = np.arange(len(v_rows))
    for i, (vk, _, vcol) in enumerate(v_rows):
        pr = bv[vk]["pre_romance_acquaintance_years"]
        mn_v = pr["mean"]
        lo_v, hi_v = pr["ci_95"]
        ax_c.barh(i, mn_v, height=0.56, color=vcol, alpha=0.88, zorder=2)
        ax_c.errorbar(mn_v, i, xerr=[[mn_v - lo_v], [hi_v - mn_v]], color=INK, fmt="none", elinewidth=0.9, capsize=1.8, zorder=4)
        ax_c.text(hi_v + 0.08, i, f"{mn_v:.2f}", va="center", ha="left", fontsize=4.9, fontweight="bold", color=INK)
    ax_c.grid(False)
    ax_c.grid(axis="x", color=GRID, lw=0.55)
    ax_c.set_yticks(yc, [r[1] for r in v_rows], fontsize=5.2)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 4.15)
    q_app = bv["online_dating_app"]["intact_relationship_quality_1_to_5"]["mean"]
    q_ins = hcm["institutional_and_kin_pooled"]["intact_relationship_quality_1_to_5"]["mean"]
    q_dif = hcm["intact_quality_boundary_check"]["online_app_minus_offline_pts_1_to_5"]
    c_rat = hcm["audit_compression_ratios"]["institutional_kin_to_app_mean_ratio"]
    ax_c.text(
        4.08, 6.42,
        f"Inst./kin \u00f7 app = {c_rat:.1f}\u00d7\nIntact qual: {q_app:.2f} vs {q_ins:.2f}\n(adj {q_dif:+.2f}, null)",
        fontsize=4.65, color=INK, ha="right", va="bottom", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_c, "C", "App witness compression (HCMST)", xlabel="Pre-romance acquaintance (yr)", fontsize=7.5)

    # Panel D: 25-Year Sternberg Triangular Love Arc + Emotional Jealousy & Option Illusion
    ax_d = fig.add_subplot(outer[1, 0])
    tla = kow["triangular_love_arc"]["by_duration"]
    d_keys = ("<=1y", "1-3y", "3-7y", "7-15y", "15-25y", "25y+")
    d_labs = ["\u22641y", "1\u20133", "3\u20137", "7\u201315", "15\u201325", "25y+"]
    dx = np.arange(len(d_keys))
    for comp, col, mk, lab in (
        ("commitment", VIOLET, "D", "Commitment"),
        ("intimacy", AQUA, "o", "Intimacy"),
        ("passion", WOMEN, "^", "Passion"),
    ):
        ys = [tla[k][comp]["mean"] for k in d_keys]
        los = [tla[k][comp]["ci_95"][0] for k in d_keys]
        his = [tla[k][comp]["ci_95"][1] for k in d_keys]
        ax_d.fill_between(dx, los, his, color=col, alpha=0.13, lw=0)
        ax_d.plot(dx, ys, color=col, lw=1.6, marker=mk, ms=3.2, label=lab)
    ax_d.set_xticks(dx, d_labs, fontsize=5.6)
    ax_d.set_ylim(2.90, 4.72)
    panel(ax_d, "D", "25-yr commitment arc (50.0k)", xlabel="Relationship duration", ylabel="Sternberg love (1\u20135)", fontsize=7.5)
    ax_d.legend(loc="upper right", fontsize=4.75, ncol=3, columnspacing=0.35, handlelength=1.0, frameon=True, facecolor="white", framealpha=0.90, edgecolor="none")
    j_em = kow["jealousy_asymmetry"]["by_sex"]["women"]["emotional_1_to_7"]["mean"]
    j_sx = kow["jealousy_asymmetry"]["by_sex"]["women"]["sexual_1_to_7"]["mean"]
    sm_inf = kow["social_media_and_alternatives"]["slopes_per_daily_hour"]["infidelity_pp_per_hour"]["estimate"]
    ax_d.text(
        0.08, 2.99,
        f"W jealousy: emot. {j_em:.2f} > sex. {j_sx:.2f}/7\nSocial media (obs.): +{sm_inf:.1f} pp/hr infid.",
        fontsize=4.65, color=INK, va="bottom", linespacing=1.04,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )

    # Panel E: 1.05M Self-Written Human Voice Essays (POV)
    ax_e = fig.add_subplot(outer[1, 1])
    v_items = [
        ("family_kids_home", "Family / home"),
        ("laugh_humor_smile", "Laughter / smile"),
        ("kind_caring_loving", "Kind / warm"),
        ("honest_loyal_trust", "Honest / loyal"),
        ("rich_six_figure", "Rich / 6-figure"),
        ("tall_6ft_mention", "6'0\" / tall guy"),
    ]
    ye = np.arange(len(v_items))
    for i, (vk, _) in enumerate(v_items):
        wv = 100.0 * swv["themes"][vk]["Female"]["share"]
        mv = 100.0 * swv["themes"][vk]["Male"]["share"]
        ax_e.plot([mv, wv], [i, i], color=INK2, lw=0.9, alpha=0.6)
        dot(ax_e, [mv], [i], MEN, "o", s=14, z=3, label="Men" if i == 0 else None)
        dot(ax_e, [wv], [i], WOMEN, "D", s=14, z=4, label="Women" if i == 0 else None)
    ax_e.grid(False)
    ax_e.grid(axis="x", color=GRID, lw=0.55)
    ax_e.set_yticks(ye, [x[1] for x in v_items], fontsize=5.5)
    ax_e.invert_yaxis()
    ax_e.set_xlim(-1.0, 34.0)
    panel(ax_e, "E", "Human voice in 1.05M essays", xlabel="Mentioned in own words (%)", fontsize=7.5)
    ax_e.legend(loc="center right", bbox_to_anchor=(1.0, 0.50), fontsize=5.0, handletextpad=0.25, labelspacing=0.16)
    r_ft = swv["voice_to_checkbox_ratios"]["female_family_to_tall"]
    st_k = kow["short_vs_long_term_ideals"]["by_sex"]["women"]["kindness"]["short_term_mean_1_to_11"]
    ax_e.text(
        29.2, 4.55, f"Family vs 6'0\": {r_ft:.0f}\u00d7;\nW ST kindness #1 ({st_k:.1f}/11)",
        fontsize=4.65, color=INK, ha="right", va="center", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )

    # Panel F: 160-Society Ethnographic Divorce Causes (Betzig SCCS)
    ax_f = fig.add_subplot(outer[1, 2])
    s_items = [
        ("adultery_infidelity", "Adultery / betrayal", VIOLET),
        ("sterility_childlessness", "Sterility", B300),
        ("cruelty_maltreatment", "Cruelty / abuse", WOMEN),
        ("laziness_economic_neglect", "Economic neglect", MEN),
        ("old_age_loss_of_looks", "Old age / lost looks", GOLD),
    ]
    yf = np.arange(len(s_items))
    for i, (sk, _, col_s) in enumerate(s_items):
        sv = 100.0 * sccs["causes"][sk]["share_mentioned"]
        ax_f.barh(i, sv, height=0.56, color=col_s, alpha=0.90)
        ax_f.text(sv + 1.1, i, f"{sv:.1f}%", va="center", ha="left", fontsize=5.1, fontweight="bold", color=INK)
    ax_f.grid(False)
    ax_f.grid(axis="x", color=GRID, lw=0.55)
    ax_f.set_yticks(yf, [x[1] for x in s_items], fontsize=5.4)
    ax_f.invert_yaxis()
    ax_f.set_xlim(0, 68)
    s_rat = sccs["adultery_to_aging_looks_ratio"]
    ax_f.text(
        66.5, 4.15, f"Betrayal \u00f7 aging\nlooks = {s_rat:.1f}\u00d7",
        fontsize=4.75, color=INK, ha="right", va="center", fontweight="bold", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_f, "F", "Divorce causes (160 SCCS soc.)", xlabel="Societies citing cause (%)", fontsize=7.5)

    return pdf(fig)


# ---------------------------------------------------------------------------
# 4C. PRIMARY VS. SECONDARY MATING CRITERIA & MALE-FEMALE LIFECYCLE DYNAMICS
# ---------------------------------------------------------------------------
def primary_secondary_dynamics_plot(data):
    speed = data["speed_dating"]
    pov = data["pov"]
    osyn = data["open_synthesis"]
    kow = osyn["kowal_2025"]
    hcm = osyn["hcmst_2017"]

    fig = plt.figure(figsize=(7.24, 5.08))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.08, 1.05], height_ratios=[1.02, 1.0], wspace=0.46, hspace=0.52)
    fig.subplots_adjust(left=0.092, right=0.988, bottom=0.100, top=0.918)

    # Panel A: Cross-Sex Mind-Reading Blindness (Perceived 2_1 minus Actual 1_1) & 4.5x Initiation Asymmetry
    ax_a = fig.add_subplot(outer[0, 0])
    csm = speed["checklist_and_hindsight"]["cross_sex_misperception"]
    init_s = speed["checklist_and_hindsight"]["post_event_followup_3_4_weeks"]["initiation_by_sex"]
    attrs_a = [
        ("attr", "Looks"),
        ("shar", "Shared int."),
        ("fun", "Fun / humor"),
        ("amb", "Ambition"),
        ("sinc", "Sincerity"),
        ("intel", "Intelligence"),
    ]
    ya = np.arange(len(attrs_a))
    ax_a.axvline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    for (ckey, clbl, col, mk), off in zip(
        (("women_about_men", "W reading M", WOMEN, "D"), ("men_about_women", "M reading W", MEN, "o")),
        (-0.15, 0.15),
    ):
        dp = csm[ckey]["delta_points"]
        ests = [dp[ak]["estimate"] for ak, _ in attrs_a]
        los = [dp[ak]["interval_95"][0] for ak, _ in attrs_a]
        his = [dp[ak]["interval_95"][1] for ak, _ in attrs_a]
        ax_a.errorbar(
            ests, ya + off, xerr=[[e - l for e, l in zip(ests, los)], [h - e for e, h in zip(ests, his)]],
            color=col, fmt=mk, ms=3.5, elinewidth=0.95, capsize=1.8, label=clbl,
        )
    ax_a.grid(False)
    ax_a.grid(axis="x", color=GRID, lw=0.55)
    ax_a.set_yticks(ya, [a[1] for a in attrs_a], fontsize=5.4)
    ax_a.set_ylim(5.6, -1.35)
    ax_a.set_xlim(-8.8, 12.2)
    ax_a.legend(loc="upper left", fontsize=4.65, ncol=2, columnspacing=0.4, handletextpad=0.2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")
    call_r = init_s["male_to_female_calls_per_match_ratio"]
    ax_a.text(
        11.8, 5.42,
        f"Post-match calls:\nMen {call_r:.1f}\u00d7 women",
        fontsize=4.65, color=INK, ha="right", va="bottom", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_a, "A", "Cross-sex mind-reading error", xlabel="Perceived \u2212 actual weight (pts)", fontsize=7.4)

    # Panel B: The Homophily Paradox (Strong Demographic Sorting vs. Dominant Primary Kindness Love)
    ax_b = fig.add_subplot(outer[0, 1])
    hvk = kow["homophily_vs_kindness"]
    rows_b = [
        ("Primary kindness", hvk["politics"]["partner_kindness_prqc_sd_per_sd"], None, AQUA),
        ("Politics match", hvk["politics"]["homophily_prqc_sd_per_sd"], hvk["politics"]["assortative_pearson_r"], VIOLET),
        ("Class match", hvk["social_class"]["homophily_prqc_sd_per_sd"], hvk["social_class"]["assortative_pearson_r"], MEN),
        ("Education match", hvk["education"]["homophily_prqc_sd_per_sd"], hvk["education"]["assortative_pearson_r"], GOLD),
    ]
    yb = np.arange(len(rows_b))
    ax_b.axvline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    for i, (_, sl_d, sort_r, col_b) in enumerate(rows_b):
        ev = sl_d["estimate"]
        lo, hi = sl_d["ci_95"]
        ax_b.barh(i, ev, height=0.52, color=col_b, alpha=0.88, zorder=2)
        ax_b.errorbar(ev, i, xerr=[[ev - lo], [hi - ev]], color=INK, fmt="none", elinewidth=0.95, capsize=1.8, zorder=4)
        tag = f"+{ev:.2f}" if sort_r is None else f"+{ev:.2f} (sort r=+{sort_r:.2f})"
        ax_b.text(hi + 0.012, i, tag, va="center", ha="left", fontsize=4.65, fontweight="bold" if sort_r is None else "normal", color=INK)
    ax_b.grid(False)
    ax_b.grid(axis="x", color=GRID, lw=0.55)
    ax_b.set_yticks(yb, [r[0] for r in rows_b], fontsize=5.3)
    ax_b.set_ylim(4.15, -0.55)
    ax_b.set_xlim(-0.02, 0.50)
    r_lo = hvk["politics"]["kindness_to_homophily_ratio"]
    r_hi = hvk["education"]["kindness_to_homophily_ratio"]
    ax_b.text(
        0.485, 4.02,
        f"Kindness = {r_lo:.1f}\u2013{r_hi:.1f}\u00d7 sorting tags",
        fontsize=4.65, color=INK, ha="right", va="bottom", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_b, "B", "Homophily paradox (150 ctry.)", xlabel="Bond love slope (PRQC SD / SD)", fontsize=7.4)

    # Panel C: Partner Reverence (mate > self) vs. The "Settling" Kink (mate < self)
    ax_c = fig.add_subplot(outer[0, 2])
    rsk = kow["reverence_vs_settling_kink"]["by_sex"]
    traits_c = [("kindness", "Kindness"), ("looks", "Looks"), ("resources", "Money")]
    yc = np.arange(len(traits_c))
    ax_c.axvline(0.0, color=MUTED, lw=0.75, ls=(0, (2, 2)))
    for (s_k, col, mk), off in zip((("women", WOMEN, "D"), ("men", MEN, "o")), (-0.15, 0.15)):
        sp = rsk[s_k]["country_fe_spline_sd_per_pt"]
        for i, (tk, _) in enumerate(traits_c):
            st_d = sp[f"settling_deficit_{tk}"]
            rv_d = sp[f"reverence_surplus_{tk}"]
            ax_c.errorbar(
                st_d["estimate"], i + off,
                xerr=[[st_d["estimate"] - st_d["ci_95"][0]], [st_d["ci_95"][1] - st_d["estimate"]]],
                color=col, fmt=mk, mfc="white", ms=3.4, elinewidth=0.9, capsize=1.6,
            )
            ax_c.errorbar(
                rv_d["estimate"], i + off,
                xerr=[[rv_d["estimate"] - rv_d["ci_95"][0]], [rv_d["ci_95"][1] - rv_d["estimate"]]],
                color=col, fmt=mk, mfc=col, ms=3.5, elinewidth=0.95, capsize=1.6,
            )
    ax_c.grid(False)
    ax_c.grid(axis="x", color=GRID, lw=0.55)
    ax_c.set_yticks(yc, [t[1] for t in traits_c], fontsize=5.5)
    ax_c.set_ylim(2.90, -1.02)
    ax_c.set_xlim(-0.22, 0.16)
    leg_c = [
        Line2D([], [], marker="o", ls="", ms=3.3, mfc="white", mec=INK, label="Settled (<self)"),
        Line2D([], [], marker="o", ls="", ms=3.4, mfc=INK, mec=INK, label="Revered (>self)"),
    ]
    ax_c.legend(handles=leg_c, loc="upper left", fontsize=4.55, ncol=2, columnspacing=0.35, handletextpad=0.15, frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")
    w_inf_drop = rsk["women"]["tiers_by_trait"]["looks"]["settled_mate_lt_self"]["infidelity_rate_pct"]["mean"] - rsk["women"]["tiers_by_trait"]["looks"]["revered_mate_gt_self"]["infidelity_rate_pct"]["mean"]
    m_inf_drop = rsk["men"]["tiers_by_trait"]["looks"]["settled_mate_lt_self"]["infidelity_rate_pct"]["mean"] - rsk["men"]["tiers_by_trait"]["looks"]["revered_mate_gt_self"]["infidelity_rate_pct"]["mean"]
    ax_c.text(
        0.152, 2.76,
        f"Revered vs settled:\n\u2212{w_inf_drop:.0f} to \u2212{m_inf_drop:.0f} pp infid.",
        fontsize=4.6, color=INK, ha="right", va="bottom", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_c, "C", "Reverence vs. settling kink", xlabel="Love slope per 1\u20137 pt (SD)", fontsize=7.4)

    # Panel D: Pre-Commitment Courtship Staging Surge (Single -> Dating -> Committed -> Married)
    ax_d = fig.add_subplot(outer[1, 0])
    stg = kow["courtship_effort_and_soi_lifecycle"]["by_relationship_stage"]["by_sex"]
    s_keys = ["single", "dating", "committed", "married"]
    s_labs = ["Single", "Dating", "Commit.", "Married"]
    sx = np.arange(4)
    for s_k, s_lbl, col, mk in (("women", "Women", WOMEN, "D"), ("men", "Men", MEN, "o")):
        means = [stg[s_k][k]["total_effort_min"]["mean"] for k in s_keys]
        los = [stg[s_k][k]["total_effort_min"]["ci_95"][0] for k in s_keys]
        his = [stg[s_k][k]["total_effort_min"]["ci_95"][1] for k in s_keys]
        ax_d.fill_between(sx, los, his, color=col, alpha=0.14, lw=0)
        ax_d.plot(sx, means, color=col, lw=1.6, marker=mk, ms=3.4, label=s_lbl)
    ax_d.set_xticks(sx, s_labs, fontsize=5.5)
    ax_d.set_xlim(-0.25, 3.25)
    ax_d.set_ylim(148, 254)
    ax_d.legend(loc="upper right", fontsize=4.9, ncol=2, columnspacing=0.45)
    dw = abs(stg["women"]["dating_to_married_drop_pct"])
    dm = abs(stg["men"]["dating_to_married_drop_pct"])
    ax_d.text(
        -0.12, 151.5,
        f"Marriage drop: \u2212{dw:.0f}% W,\n\u2212{dm:.0f}% M (Prop. 23i)",
        fontsize=4.65, color=INK, ha="left", va="bottom", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_d, "D", "Courtship staging surge (65.9k)", xlabel="Relationship stage (obs.)", ylabel="Daily effort (min/day)", fontsize=7.4)

    # Panel E: Pre-Cohabitation Rush Hazard & Marriage-Specific Female Unilateral Exit Flip (HCMST)
    ax_e = fig.add_subplot(outer[1, 1])
    crh = hcm["cohabitation_rush_hazard"]
    bw = crh["by_romance_to_cohab_window"]
    bia = hcm["breakup_initiation_asymmetry"]
    w_keys = ["le_3m_rush", "gt_3m_to_1y", "gt_1y_to_2_5y", "gt_2_5y"]
    w_labs = ["\u22643 mo", "3\u201312m", "1\u20132.5y", "2.5y+"]
    w_cols = [RED, GOLD, B300, AQUA]
    wx = np.arange(4)
    for i, (wk, wcol) in enumerate(zip(w_keys, w_cols)):
        sh = 100.0 * bw[wk]["breakup_share"]
        lo, hi = 100.0 * bw[wk]["wilson_ci_95"][0], 100.0 * bw[wk]["wilson_ci_95"][1]
        ax_e.bar(i, sh, width=0.58, color=wcol, alpha=0.86, zorder=2)
        ax_e.errorbar(i, sh, yerr=[[sh - lo], [hi - sh]], color=INK, fmt="none", elinewidth=0.95, capsize=2.0, zorder=4)
        ax_e.text(i, hi + 0.35, f"{sh:.1f}%", ha="center", va="bottom", fontsize=4.8, fontweight="bold", color=INK)
    ax_e.grid(False)
    ax_e.grid(axis="y", color=GRID, lw=0.55)
    ax_e.set_xticks(wx, w_labs, fontsize=5.5)
    ax_e.set_xlim(-0.55, 3.55)
    ax_e.set_ylim(0, 14.8)
    div_w = 100.0 * bia["married_divorce_or_separation"]["woman_share_of_unilateral"]
    hus_w = 100.0 * bia["married_divorce_or_separation"]["by_respondent_gender_unilateral"]["male_respondents_wife_share"]
    ax_e.text(
        3.45, 14.1,
        f"Rush = {crh['le_3m_to_gt_2_5y_breakup_ratio']:.1f}\u00d7 breakup;\nDivorce: {div_w:.0f}% W ({hus_w:.0f}% M-rep)",
        fontsize=4.6, color=INK, ha="right", va="top", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_e, "E", "Rushed cohab & exit flip (HCMST)", xlabel="Pre-cohabitation wait", ylabel="Observed breakup (%)", fontsize=7.4)

    # Panel F: Lifespan Hearth Action-Verb Calendar & Widowhood vs. Divorce Second-Act Re-Entry (1.14M POV)
    ax_f = fig.add_subplot(outer[1, 2])
    avc = pov["self_written_voice"]["action_verb_calendar"]
    sar = pov["second_act_reentry"]["headline_contrasts"]
    a_br = [("18-22", 20.0), ("23-27", 25.0), ("28-34", 31.0), ("35-44", 39.5), ("45-54", 49.5), ("55-64", 59.5), ("65-79", 71.0)]
    ax_x = np.array([x for _, x in a_br])
    w_ck = [100.0 * avc["by_age_bracket"][k]["cook_bake_hearth"]["Female"]["share"] for k, _ in a_br]
    m_ck = [100.0 * avc["by_age_bracket"][k]["cook_bake_hearth"]["Male"]["share"] for k, _ in a_br]
    p_pt = [100.0 * avc["by_age_bracket"][k]["party_nightlife"]["Pooled"]["share"] for k, _ in a_br]
    sh_v = avc["lifespan_shifts_18_22_to_65_79"]
    ax_f.plot(ax_x, w_ck, color=WOMEN, lw=1.6, marker="D", ms=3.1, label=f"W cook/hearth ({sh_v['female_cook_rise_ratio']:.1f}\u00d7)")
    ax_f.plot(ax_x, m_ck, color=MEN, lw=1.6, marker="o", ms=3.1, label=f"M cook/hearth ({sh_v['male_cook_rise_ratio']:.1f}\u00d7)")
    ax_f.plot(ax_x, p_pt, color=GOLD, lw=1.35, ls="--", marker="^", ms=3.0, label=f"Party/club ({sh_v['pooled_party_drop_pct']:.0f}%)")
    ax_f.set_xlim(17, 74)
    ax_f.set_ylim(0, 39.0)
    ax_f.set_xticks([20, 35, 50, 65])
    ax_f.legend(loc="upper left", fontsize=4.55, labelspacing=0.14, handlelength=1.2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="none")
    dm_drop = abs(sar["divorced_vs_single_female_marry_change_pct_40_49"])
    ax_f.text(
        18.2, 20.0,
        f"40\u201349 remarry: Div. W \u2212{dm_drop:.0f}%\n(81% serious); Wid. M > W",
        fontsize=4.55, color=INK, ha="left", va="bottom", linespacing=1.02,
        bbox=dict(boxstyle="round,pad=0.12", fc=BAND, ec="none", alpha=0.92),
    )
    panel(ax_f, "F", "Hearth verbs & 2nd act (1.05M)", xlabel="Age (years)", ylabel="Mentions in essays (%)", fontsize=7.4)

    return pdf(fig)


# ---------------------------------------------------------------------------
# 5. 6.68M HON ARCHITECTURE: LORENZ ATTENTION & PORTFOLIO (Gate G14 deduplicated)
# ---------------------------------------------------------------------------
def hon_architecture_plot(data):
    hon = data["hon"]
    att = hon["attention_and_portfolio"]

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.32), gridspec_kw={"width_ratios": [1.02, 1.08]})
    fig.subplots_adjust(left=0.088, right=0.925, bottom=0.205, top=0.845, wspace=0.36)

    # Panel A: Voter Attention Inequality (Lorenz Curves & Gini)
    ax = axes[0]
    pop_x = np.linspace(0, 1, 11)
    ax.plot(pop_x, pop_x, color=MUTED, lw=0.75, ls=(0, (2, 2)), label="Equal attention")
    for sex, label, col, mk in SERIES:
        s_att = att["by_sex"][sex]
        cum_v = np.r_[0.0, np.cumsum(s_att["lorenz_deciles_votes"])]
        cum_l = np.r_[0.0, np.cumsum(s_att["lorenz_deciles_likes"])]
        ax.plot(
            pop_x, cum_v, color=col, lw=1.55, marker=mk, ms=2.8,
            label=f"{label} votes (Gini {s_att['votes_gini']:.2f}, med {s_att['median_votes']:,.0f})",
        )
        ax.plot(pop_x, cum_l, color=col, lw=1.15, ls="--", alpha=0.78, label=f"{label} likes (Gini {s_att['likes_gini']:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel(ax, "A", "18.5\u00d7 voter skew & attention concentration (6.68M)", xlabel="Cumulative share of profiles", ylabel="Cumulative share of votes / likes", percent=True)
    ax.legend(loc="upper left", fontsize=5.3, labelspacing=0.18)

    # Panel B: Portfolio Image Count Buys Voter Volume, Not Rank
    ax = axes[1]
    pkeys = ("0", "1", "2", "3", "4", "5_plus")
    plabs = ["0", "1", "2", "3", "4", "5+"]
    px = np.arange(len(pkeys))
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(MUTED)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.tick_params(labelsize=6.5, colors=INK2)
    for sex, label, col, mk in SERIES:
        by_p = att["by_photo_count"][sex]
        ranks = [100 * by_p[k]["mean_within_sex_percentile"] for k in pkeys]
        v_rel = [by_p[k]["median_votes"] / by_p["1"]["median_votes"] for k in pkeys]
        ax.plot(px, ranks, color=col, lw=1.6, marker=mk, ms=3.4, label=f"{label} beauty pct. (left)")
        ax2.plot(px, v_rel, color=col, lw=1.25, ls="--", marker=mk, mfc="white", ms=3.2)
    ax.set_xticks(px, plabs)
    ax.set_ylim(18, 69)
    ax2.set_ylim(0.50, 1.85)
    ax2.set_ylabel("Median votes vs. 1 photo (dashed)", fontsize=6.8, color=INK2)
    panel(ax, "B", "Extra photos raise votes +50%, not beauty rank", xlabel="Inventoried photos per profile", ylabel="Mean beauty percentile (solid)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, 0.99), fontsize=5.6, labelspacing=0.18)

    return pdf(fig)


# ---------------------------------------------------------------------------
# 6. HOW MEN AND WOMEN CHOOSE ACROSS ADULTHOOD, AGES 18-75 (Fig. 14)
# ---------------------------------------------------------------------------
def pov_lifecourse_plot(data):
    pov = data["pov"]
    hon_b = data["hon"]["beauty_by_age"]
    fa_age = data["facial_attribution"]["age_bucket_progression"]
    osyn = data["open_synthesis"]
    kow = osyn["kowal_2025"]
    wal = osyn["walter_2020"]["age_gap_by_age"]
    whyte_h = osyn["whyte_2021"]["happiness_by_partnered"]
    bands = [(lo, hi) for lo, hi in data["cohorts"]["pov_bands"] if hi <= 74]

    fig, axes = plt.subplots(2, 3, figsize=(7.24, 4.95))
    fig.subplots_adjust(left=0.076, right=0.99, bottom=0.105, top=0.905, wspace=0.40, hspace=0.49)

    def _shared_age(ax):
        ax.set_xlim(17, 76)
        ax.set_xticks([20, 35, 50, 65, 75])
        ax.set_xlabel("Age (years)", labelpad=3.0)

    # ROW 1: DIVERGING BIOLOGICAL & MARKET CLOCKS (AGES 18-75)
    # Panel A: Realized Partner Age-Gap Scissors (Kowal 37.7k + Walter 46.9k)
    ax = axes[0, 0]
    k_br = [("18-24", 21.0), ("25-29", 27.0), ("30-34", 32.0), ("35-39", 37.0), ("40-49", 44.5), ("50-59", 54.5), ("60-79", 68.0)]
    kx = np.array([x for _, x in k_br])
    pag = kow["partner_age_gap_by_age"]["brackets"]
    kw_y = np.array([pag[k]["women"]["mean"] for k, _ in k_br])
    km_y = np.array([pag[k]["men"]["mean"] for k, _ in k_br])
    ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.fill_between(kx, km_y, kw_y, color=WOMEN, alpha=0.12, lw=0)
    ax.plot(kx, kw_y, color=WOMEN, lw=1.65, marker="D", ms=3.2, label="Women (Kowal)")
    ax.plot(kx, km_y, color=MEN, lw=1.65, marker="o", ms=3.2, label="Men (Kowal)")
    w_br = [("18-24", 21.0), ("25-29", 27.0), ("30-39", 34.5), ("40-49", 44.5), ("50-75", 61.0)]
    wx_w = [x for k, x in w_br if k in wal]
    ax.scatter(wx_w, [wal[k]["women"]["mean"] for k, _ in w_br if k in wal], s=18, marker="D", facecolor="white", edgecolor=WOMEN, lw=1.0, zorder=4)
    ax.scatter(wx_w, [wal[k]["men"]["mean"] for k, _ in w_br if k in wal], s=20, marker="o", facecolor="white", edgecolor=MEN, lw=1.0, zorder=4, label="Walter 45-ctry")
    _shared_age(ax)
    ax.set_ylim(-6.6, 4.2)
    panel(ax, "A", "Partner age scissors (84.6k)", ylabel="Partner age minus own (yr)")
    ax.legend(loc="lower left", fontsize=5.3, labelspacing=0.16)
    gap60 = pag["60-79"]["women_minus_men_years"]
    ax.text(74.5, -0.55, f"60+: {gap60:.1f}-yr\nscissors", fontsize=5.2, color=INK, ha="right", va="top", fontweight="bold", linespacing=1.0)

    # Panel B: Platform Voter Rank vs. Pure Facial Anatomy (6.68M HON + 5,353 Aligned Faces)
    ax = axes[0, 1]
    ages = np.arange(18, 76)
    yw_h = np.array([100.0 * hon_b[str(a)]["Female"]["all"]["mean_percentile"] for a in ages])
    ym_h = np.array([100.0 * hon_b[str(a)]["Male"]["all"]["mean_percentile"] for a in ages])
    ax.axhline(50, color=MUTED, lw=0.7, ls=(0, (2, 2)))
    ax.fill_between(ages, ym_h, yw_h, where=(yw_h >= ym_h), interpolate=True, color=WOMEN, alpha=0.12, lw=0)
    ax.fill_between(ages, ym_h, yw_h, where=(ym_h >= yw_h), interpolate=True, color=MEN, alpha=0.13, lw=0)
    ax.plot(ages, yw_h, color=WOMEN, lw=1.6, label="Women voter rank")
    ax.plot(ages, ym_h, color=MEN, lw=1.6, label="Men voter rank")
    fa_x = [21.0, 29.5, 39.5, 49.5, 62.0]
    fa_k = ["18-24", "25-34", "35-44", "45-54", "55+"]
    ax.plot(fa_x, [fa_age["Female"][k]["mean_predicted_pct"] for k in fa_k], color=WOMEN, lw=1.25, ls="--", marker="D", mfc="white", ms=3.2, label="W face-only")
    ax.plot(fa_x, [fa_age["Male"][k]["mean_predicted_pct"] for k in fa_k], color=MEN, lw=1.25, ls="--", marker="o", mfc="white", ms=3.2, label="M face-only")
    _shared_age(ax)
    ax.set_ylim(3, 86)
    panel(ax, "B", "Photo rank vs. face (6.68M)", ylabel="Within-sex percentile")
    ax.legend(loc="lower left", fontsize=5.0, ncol=2, columnspacing=0.40, handlelength=1.25, labelspacing=0.14)

    # Panel C: Childless 'Want Children' Crossover at Age ~42 (1.14M POV + 5/5 Countries)
    ax = axes[0, 2]
    v_bands = [
        (lo, hi) for lo, hi in bands
        if pov["bands"][f"{lo}-{hi}"]["Female"]["without_children"]
        and pov["bands"][f"{lo}-{hi}"]["Male"]["without_children"]
    ]
    centers = np.array([0.5 * (lo + hi) for lo, hi in v_bands])
    yw_c = np.array([100.0 * pov["bands"][f"{lo}-{hi}"]["Female"]["without_children"]["shares"]["want"] for lo, hi in v_bands])
    ym_c = np.array([100.0 * pov["bands"][f"{lo}-{hi}"]["Male"]["without_children"]["shares"]["want"] for lo, hi in v_bands])
    ax.fill_between(centers, ym_c, yw_c, where=(yw_c >= ym_c), interpolate=True, color=WOMEN, alpha=0.13, lw=0)
    ax.fill_between(centers, ym_c, yw_c, where=(ym_c >= yw_c), interpolate=True, color=MEN, alpha=0.15, lw=0)
    ax.plot(centers, yw_c, color=WOMEN, marker="D", ms=3.0, lw=1.6, label="Women")
    ax.plot(centers, ym_c, color=MEN, marker="o", ms=3.0, lw=1.6, label="Men")
    xc = pov["childless_crossover"]["band_midpoint_interpolated_age"]
    ax.axvline(xc, color=VIOLET, lw=0.9, ls=":")
    ax.scatter([xc], [np.interp(xc, centers, yw_c)], s=22, facecolor=VIOLET, edgecolor="white", lw=0.5, zorder=5)
    _shared_age(ax)
    ax.set_ylim(0, 56)
    panel(ax, "C", "Childless 'Want kids' (~42)", ylabel="Childless wanting children (%)")
    ax.text(xc + 1.5, 42.0, f"Crosses age {xc:.1f}\n(5/5 countries:\nUS, UK, CA, BR, AU)", fontsize=5.1, color=VIOLET, fontweight="bold", linespacing=1.05)

    # ROW 2: UNBROKEN HUMAN TRUTHS ACROSS ADULTHOOD (AGES 18-75)
    # Panel D: Kindness Ranks #1 at Every Age while Resources Stay Modest (67.7k Kowal)
    ax = axes[1, 0]
    b_age = kow["budget_by_age"]
    yw_knd = np.array([b_age[k]["items"]["kindness"]["women"]["mean"] for k, _ in k_br])
    ym_knd = np.array([b_age[k]["items"]["kindness"]["men"]["mean"] for k, _ in k_br])
    yw_res = np.array([b_age[k]["items"]["resources"]["women"]["mean"] for k, _ in k_br])
    ym_res = np.array([b_age[k]["items"]["resources"]["men"]["mean"] for k, _ in k_br])
    ax.fill_between(kx, ym_res, yw_res, color=GOLD, alpha=0.15, lw=0)
    ax.plot(kx, yw_knd, color=WOMEN, lw=1.65, marker="D", ms=3.1, label="Kindness (W #1)")
    ax.plot(kx, ym_knd, color=MEN, lw=1.65, marker="o", ms=3.1, label="Kindness (M #1)")
    ax.plot(kx, yw_res, color=WOMEN, lw=1.3, ls="--", marker="D", mfc="white", ms=2.9, label="Money (W)")
    ax.plot(kx, ym_res, color=MEN, lw=1.3, ls="--", marker="o", mfc="white", ms=2.9, label="Money (M)")
    _shared_age(ax)
    ax.set_ylim(2.05, 7.65)
    panel(ax, "D", "Kindness #1 at every age (67.7k)", ylabel="Points allocated (of 30)")
    ax.legend(loc="lower left", fontsize=5.0, ncol=2, columnspacing=0.45, handlelength=1.25, labelspacing=0.15)

    # Panel E: Serious Commitment Stays Unbroken Women > Men at Every Age (1.14M POV)
    ax = axes[1, 1]
    wins = ("18-24", "25-29", "30-39", "40-49", "50-59", "60-69")
    wx = np.array([21.0, 27.0, 34.5, 44.5, 54.5, 64.5])
    yw_s = np.array([100.0 * pov["market_and_commitment"][w]["by_sex"]["Female"]["serious_commitment"]["share"] for w in wins])
    ym_s = np.array([100.0 * pov["market_and_commitment"][w]["by_sex"]["Male"]["serious_commitment"]["share"] for w in wins])
    ax.fill_between(wx, ym_s, yw_s, color=WOMEN, alpha=0.14, lw=0)
    ax.plot(wx, yw_s, color=WOMEN, marker="D", ms=3.3, lw=1.65, label="Women: serious")
    ax.plot(wx, ym_s, color=MEN, marker="o", ms=3.3, lw=1.65, label="Men: serious")
    yw_10 = np.array([100.0 * pov["market_and_commitment"][w]["by_sex"]["Female"]["over_10y_union"]["share"] for w in wins])
    ym_10 = np.array([100.0 * pov["market_and_commitment"][w]["by_sex"]["Male"]["over_10y_union"]["share"] for w in wins])
    ax.plot(wx, 0.5 * (yw_10 + ym_10), color=AQUA, lw=1.3, ls=":", marker="s", ms=2.7, label="Prior >10-yr union")
    _shared_age(ax)
    ax.set_ylim(0, 90)
    panel(ax, "E", "Serious devotion stays W > M (1.14M)", ylabel="Share of profiles (%)")
    ax.legend(loc="lower right", fontsize=5.3, labelspacing=0.18)

    # Panel F: Partnered Happiness Lift (Whyte) & Education Postponement Without Desire Loss (POV)
    ax = axes[1, 2]
    edu_cats = ("High school", "Some college", "Bachelors Degree", "Masters Degree", "PhD / Post Doctoral")
    edu_short = ["HS", "Some\ncol.", "BA", "MA", "PhD"]
    ex = np.arange(len(edu_cats))
    edu_d = pov["traits_and_postponement"]["education_ages_30_49"]
    w_par = [100.0 * edu_d[c]["Female"]["parenthood"]["share"] for c in edu_cats]
    w_want = [100.0 * edu_d[c]["Female"]["childless_want_children"]["share"] for c in edu_cats]
    m_want = [100.0 * edu_d[c]["Male"]["childless_want_children"]["share"] for c in edu_cats]
    ax.plot(ex, w_par, color=VIOLET, lw=1.6, marker="s", ms=3.1, label="W already parent")
    ax.plot(ex, w_want, color=WOMEN, lw=1.5, ls="--", marker="D", mfc="white", ms=3.1, label="W childless want")
    ax.plot(ex, m_want, color=MEN, lw=1.5, ls="--", marker="o", mfc="white", ms=3.1, label="M childless want")
    ax.set_xticks(ex, edu_short, fontsize=6.0)
    ax.set_ylim(12, 88)
    panel(ax, "F", "Postponement & happiness lift", xlabel="Schooling (ages 30–49)", ylabel="Share of profiles (%)")
    ax.legend(loc="upper right", fontsize=5.1, labelspacing=0.16, handlelength=1.3)
    hw = whyte_h["Women"]["raw_lift_pts"]["estimate"]
    hm = whyte_h["Men"]["raw_lift_pts"]["estimate"]
    ax.text(
        0.05, 16.5,
        f"Partnered happiness (N=7,060):\nW +{hw:.1f} pts, M +{hm:.1f} pts (0–100)",
        fontsize=5.0, color=INK, va="bottom", linespacing=1.06,
        bbox=dict(boxstyle="round,pad=0.15", fc=BAND, ec="none", alpha=0.92),
    )

    return pdf(fig)


# ---------------------------------------------------------------------------
# 7. MULTIMODAL BENCHMARK CEILING & MHC NULL SYNTHESIS (performance-fig-ai-benchmark.pdf)
# ---------------------------------------------------------------------------
def ai_benchmark_plot(data):
    bm = data["inventory"]["multimodal_benchmark"]["waves"]
    mhc = data["open_synthesis"]["published_benchmarks"]["havlicek_2020_mhc"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.20, 2.75), gridspec_kw={"width_ratios": [1.32, 1.0]})
    fig.subplots_adjust(left=0.08, right=0.988, bottom=0.20, top=0.83, wspace=0.38)

    # Panel A: Multi-Wave H1-H24 Multimodal Benchmark Progression & Static-Context Ceiling
    xs = np.arange(1, len(bm) + 1)
    rs = [w["profile_test"]["spearman_r"] for w in bm]
    r2s = [r ** 2 for r in rs]
    peak_r = max(rs)
    peak_r2 = max(r2s)
    ax.axhspan(peak_r, 1.0, color=BAND, zorder=0, lw=0)
    ax.plot(xs, rs, color=MEN, lw=1.7, marker="o", ms=3.2, label="Test Spearman $\\rho$ (rank)")
    ax.plot(xs, r2s, color=VIOLET, lw=1.35, ls="--", marker="s", ms=2.6, label="Explained rank variance $\\rho^2$")
    ticks_x = [1, 4, 7, 10, 15, 20] + ([len(bm)] if len(bm) > 20 else [])
    ticks_l = ["H1\nRaw UV", "H4\nSFace", "H7\nPct-CDF", "H10\nFiLM+GBDT", "H15\nMulti-img", "H20\nEnsemble"] + ([f"H{len(bm)}\n3D/Spec"] if len(bm) > 20 else [])
    ax.set_xticks(ticks_x, ticks_l, fontsize=5.8)
    ax.set_ylim(0, 1.0)
    panel(ax, "A", f"{len(bm)}-wave HON facial benchmark (20,000 images)", xlabel="Progressive nuisance removal, multi-photo pooling & alignment", ylabel="Test association with human jury")
    ax.legend(loc="lower right", fontsize=6.2)
    ax.text(len(bm) * 0.50, 0.83, f"Unexplained human-jury variance\n$1 - \\rho^2 = {100*(1 - peak_r2):.1f}\\%$ above H20 ($\\rho = {peak_r:.3f}$)",
            fontsize=6.1, color=INK2, ha="center", linespacing=1.15)

    # Panel B: Havlicek et al. (2020) MHC Meta-Analytic Null Intervals
    bx.axvline(0, color=INK2, lw=0.8, ls=(0, (2, 2)))
    labels_mhc = [
        f"Genomic pairing\n(k={mhc[0]['k_effects']})",
        f"Odor preference\n(k={mhc[1]['k_effects']})",
        f"Sexual satisfaction\n(k={mhc[2]['k_effects']})",
        f"Combined synthesis\n(k={mhc[3]['k_effects']})",
    ]
    for i, row in enumerate(mhc):
        est = row["fisher_z"]
        lo, hi = row["ci_95"]
        bx.errorbar(est, i, xerr=[[est - lo], [hi - est]], color=INK, fmt="D", mfc=GOLD if i < 3 else VIOLET,
                    mec=INK, ms=5.0, elinewidth=1.25, capsize=3)
    bx.grid(False)
    bx.grid(axis="x", color=GRID, linewidth=0.55)
    bx.set_yticks(range(len(mhc)), labels_mhc, fontsize=6.2)
    bx.set_xlim(-0.22, 0.18)
    panel(bx, "B", "MHC dissimilarity meta-means include zero", xlabel="Random-effects Fisher $z$ (Havlicek et al. 2020)")

    return pdf(fig)


# ---------------------------------------------------------------------------
# 8. BACKWARD-COMPATIBLE SUPPLEMENT FIGURES (Upgraded to Unified Palette)
# ---------------------------------------------------------------------------
def pov_plot(pov, bands):
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.65))
    fig.subplots_adjust(left=0.083, right=0.993, bottom=0.20, top=0.77, wspace=0.24)
    centers = [(lo + hi) / 2 for lo, hi in bands]
    handles = []
    for ax, letter, heading, key in zip(axes, "ABC", ("Reported parenthood", "Without children", "With children"),
                                        ("parenthood", "without_children", "with_children")):
        panel(ax, letter, heading, ylabel="Share of profiles" if letter == "A" else None, percent=True)
        age_axis(ax)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1])
        for lo, hi in ((25, 29), (45, 49)):
            if letter == "B":
                ax.axvspan(lo - 0.5, hi + 0.5, color=BAND, linewidth=0, zorder=0)
        for sex, label, color, marker in SERIES:
            cells = [pov["bands"][f"{lo}-{hi}"][sex][key] for lo, hi in bands]
            ys = [(cell["share"] if key == "parenthood" else cell["shares"]["want"]) if cell else math.nan for cell in cells]
            line, = ax.plot(centers, ys, color=color, marker=marker, markersize=3, lw=1.5, label=label)
            if letter == "A":
                handles.append(line)
    legend_top(fig, handles, y=0.99)
    return pdf(fig)



def beauty_plot(beauty):
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.70))
    fig.subplots_adjust(left=0.083, right=0.994, bottom=0.20, top=0.79, wspace=0.29)
    ages = range(18, 80)
    for ax, letter, (sex, title, color, marker) in zip(axes[:2], "AB", SERIES):
        panel(ax, letter, title, ylabel="Beauty percentile" if letter == "A" else None)
        age_axis(ax)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        cells = [beauty[str(age)][sex]["all"] for age in ages]
        lower, upper = ([100 * cell["percentile_iqr"][i] if cell else math.nan for cell in cells] for i in (0, 1))
        ax.fill_between(ages, lower, upper, color=color, alpha=0.16, lw=0)
        ax.axhline(50, color=MUTED, lw=0.7, ls=(0, (3, 3)))
        ax.plot(ages, [100 * c["mean_percentile"] if c else math.nan for c in cells], color=color, lw=1.65,
                marker=marker, markevery=5, markersize=3)
    ax = axes[2]
    panel(ax, "C", "In the top tenth", ylabel="Share of profiles", percent=True)
    age_axis(ax)
    ax.axhline(0.10, color=MUTED, lw=0.7, ls=(0, (3, 3)))
    handles = []
    for sex, label, color, marker in SERIES:
        cells = [beauty[str(age)][sex]["all"] for age in ages]
        line, = ax.plot(ages, [c["top_tenth_n"] / c["n"] if c else math.nan for c in cells],
                        color=color, lw=1.6, marker=marker, markevery=5, markersize=3, label=label)
        handles.append(line)
    ax.set_ylim(bottom=0)
    ax.legend(handles=handles, loc="upper right", fontsize=6.8, handlelength=1.4)
    return pdf(fig)


def rating_plot(ratings):
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.80))
    fig.subplots_adjust(left=0.095, right=0.989, bottom=0.19, top=0.77, wspace=0.27)
    ages = range(18, 80)
    handles = [Line2D([], [], color=color, marker=marker, lw=1.35, markersize=3, label=label)
               for _, label, color, marker in SERIES]
    handles += [Line2D([], [], color=MUTED, linestyle=style, lw=1.15, label=label) for style, label in
                (("-", "All eligible"), ("--", "Image inventoried"))]
    for ax, letter, title, key in zip(axes, "AB", ("Profile like rates", "Recorded voting activity"),
                                      ("mean_profile_like_rate", "median_votes")):
        panel(ax, letter, title, ylabel="Mean likes per vote" if letter == "A" else "Median votes per profile",
              percent=letter == "A")
        age_axis(ax)
        if letter == "B":
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_locator(MultipleLocator(0.02))
        for sex, _, color, marker in SERIES:
            for coverage, style in (("all", "-"), ("inventoried", "--")):
                cells = [ratings[str(a)][sex][coverage] for a in ages]
                ax.plot(ages, [cell[key] if cell else math.nan for cell in cells], color=color, ls=style,
                        lw=1.4, marker=marker, markevery=7, markersize=2.8)
    legend_top(fig, handles, y=0.99)
    return pdf(fig)



def choice_plot(speed):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.20, 2.95), gridspec_kw={"width_ratios": [1, 1.05]})
    fig.subplots_adjust(left=0.09, right=0.991, bottom=0.20, top=0.79, wspace=0.42)
    panel(ax, "A", "Who said yes?", xlabel="Partner's looks rank among dates", ylabel="Share saying yes", percent=True)
    ax.set_xticks(range(1, 6), ["Bottom", "2", "3", "4", "Top"])
    ax.set_ylim(0, 0.9)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    panel(bx, "B", "Sensitivity to attractiveness", xlabel="Change in yes rate (percentage points)")
    bx.grid(False)
    bx.grid(axis="x", color=GRID, linewidth=0.55)
    bx.set_yticks([0, 1], ["Other raters", "Chooser's own"])
    bx.set_ylim(-0.55, 1.55)
    bx.set_xlim(0, 28)
    bx.axvline(0, color=GRID, lw=0.7)
    handles = []
    for (sex, label, color, marker), offset in zip(SERIES, (-0.105, 0.105)):
        name = label.lower()
        cells = speed["yes_by_panel_quintile"][name]["quintiles"]
        y = [cell["estimate"] for cell in cells]
        low, high = ([cell["interval_95"][i] for cell in cells] for i in (0, 1))
        line = ax.errorbar([i + offset for i in range(1, 6)], y, yerr=[[p - l for p, l in zip(y, low)],
                            [h - p for p, h in zip(y, high)]], color=color, marker=marker, ms=3.9,
                           lw=1.35, elinewidth=0.8, capsize=2, label=f"{label} choosing")
        handles.append(line)
        for row, method in enumerate(("panel", "own")):
            cell = speed["adjusted_slopes"][method][name]["coefficients_pp"]["attr"]
            estimate = cell["estimate"]
            low, high = cell["interval_95"]
            bx.errorbar(estimate, row + offset, xerr=[[estimate - low], [high - estimate]], color=color,
                        fmt=marker, ms=4.3, elinewidth=1, capsize=2.5)
    legend_top(fig, handles, y=0.99)
    return pdf(fig)


# ---------------------------------------------------------------------------
# 8B. DEEP-LEARNING FACIAL ATTRIBUTION & SUPER-RESOLVED CANONICAL COMPOSITES
# ---------------------------------------------------------------------------
def _load_face_npz(data):
    npz_path = ROOT / "evidence/reference/performance-face-composites.npz"
    raw = npz_path.read_bytes()
    expected = data["facial_attribution"]["composites_npz_sha256"]
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == expected, f"Composites NPZ SHA-256 mismatch: {actual} != {expected}"
    return np.load(BytesIO(raw))


def _studio_mask_384(npz_obj):
    sil = npz_obj["silhouette_384"].astype(np.float32) / 255.0
    oval_128 = npz_obj["oval_128"].astype(np.float32)
    oval_384 = np.clip(zoom(oval_128, 3.0, order=3), 0.0, 1.0)
    # Soft studio portrait vignette preserving full hairline, ears/cheeks, and chin contour
    comb = np.minimum(np.maximum(sil, oval_384 * 0.92), np.clip(oval_384 * 1.18, 0.0, 1.0))
    comb = gaussian_filter(comb, sigma=4.0)
    return np.clip((comb - 0.06) / 0.50, 0.0, 1.0) ** 1.12


def _studio_portrait(rgb_u8, mask_384, bg_rgb=(0.969, 0.973, 0.980)):
    rgb = rgb_u8.astype(np.float32) / 255.0
    bg = np.array(bg_rgb, dtype=np.float32)[None, None, :]
    out = mask_384[..., None] * rgb + (1.0 - mask_384[..., None]) * bg
    return np.clip(out, 0.0, 1.0)


def _studio_heatmap(base_rgb_u8, map_128, mask_384, vmax, bg_rgb=(0.969, 0.973, 0.980)):
    rgb = base_rgb_u8.astype(np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Subtle micro-edge relief so eyes, brows, nose, philtrum, and vermilion remain crystal clear
    lum_hi = lum + 0.35 * (lum - gaussian_filter(lum, sigma=2.5))
    relief = np.clip(0.30 + 0.66 * lum_hi, 0.0, 1.0)
    alabaster = np.stack([relief * 0.965, relief * 0.972, relief * 0.988], axis=-1)

    m384 = zoom(map_128.astype(np.float32), 3.0, order=3)
    norm = np.clip(m384 / max(float(vmax), 1e-4), -1.0, 1.0) * mask_384
    heat_rgb = FACE_CMAP((norm + 1.0) * 0.5)[..., :3]
    heat_alpha = (np.abs(norm) ** 0.76) * 0.84 * mask_384
    blended = (1.0 - heat_alpha[..., None]) * alabaster + heat_alpha[..., None] * heat_rgb
    bg = np.array(bg_rgb, dtype=np.float32)[None, None, :]
    out = mask_384[..., None] * blended + (1.0 - mask_384[..., None]) * bg
    return np.clip(out, 0.0, 1.0), m384 * mask_384


def _style_face_ax(ax, title_top=None, badge_bot=None, border_color="#D6D5CE"):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(border_color)
        sp.set_linewidth(0.6)
    if title_top:
        ax.set_title(title_top, loc="center", fontsize=6.9, fontweight="bold", pad=3.0, color=INK)
    if badge_bot:
        ax.text(
            0.5,
            0.028,
            badge_bot,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=INK,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#D8D7D2", "alpha": 0.88, "lw": 0.4},
        )


def _short_zone_label(zk):
    mapping = {
        "eyes_iris": "Eyes & aperture",
        "brows": "Eyebrow arch",
        "tear_trough": "Tear trough",
        "lips_vermilion": "Vermilion lips",
        "nose_alar_tip": "Nasal tip & alar",
        "nose_bridge": "Nasal bridge",
        "malar_cheek": "Malar cheekbone",
        "nasolabial": "Nasolabial fold",
        "jaw_chin": "Jawline & chin",
        "philtrum": "Philtrum",
        "forehead_glabella": "Forehead",
        "lateral_skin": "Lateral cheek skin",
    }
    return mapping.get(zk, zk)


def face_drivers_plot(data):
    fa = data["facial_attribution"]
    npz = _load_face_npz(data)
    mask_384 = _studio_mask_384(npz)
    canon_pts = npz["canon106_port"].astype(np.float32)
    tris = npz["interior_tris"].astype(np.int32)

    fig = plt.figure(figsize=(7.35, 7.60))
    outer = fig.add_gridspec(4, 1, height_ratios=[2.34, 0.08, 0.16, 1.52], hspace=0.18)
    top_gs = outer[0].subgridspec(2, 6, wspace=0.075, hspace=0.28)
    cbar_gs = outer[1].subgridspec(1, 3, width_ratios=[0.20, 0.60, 0.20])
    bot_gs = outer[3].subgridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.42)

    # Determine unified diverging scale across tier maps
    tier_vals = []
    geom_vals = []
    for g in ("Female", "Male"):
        for tk in ("top15", "bot15"):
            tier_vals.append(np.percentile(np.abs(npz[f"map__tier__{g}__{tk}"].astype(np.float32)), 98.5))
        geom_vals.append(np.percentile(np.abs(npz[f"map__geom_diff__{g}"].astype(np.float32)), 98.5))
    vmax_tier = max(float(max(tier_vals)), 1.5)
    vmax_geom = max(float(max(geom_vals)), 0.8)

    col_titles = [
        "Bottom 15%\ncomposite",
        "Median (42–58th)\ncomposite",
        "Top 15%\ncomposite",
        "Beauty rejectors\n(Bottom 15% map)",
        "Beauty drivers\n(Top 15% map)",
        f"{fa['delaunay_triangles']}-tri geometry\n(Top − Bottom)",
    ]

    for r_idx, (g, g_label, g_color) in enumerate((("Female", "A   Women", WOMEN), ("Male", "B   Men", MEN))):
        zd = fa["zone_drivers_and_rejectors"][g]
        mid_rgb = npz[f"comp__tier__{g}__mid"]
        bot_rgb = npz[f"comp__tier__{g}__bot15"]
        top_rgb = npz[f"comp__tier__{g}__top15"]

        portraits = [
            (bot_rgb, f"n = {zd['bot15_n']:,} | ≤15th pct"),
            (mid_rgb, f"n = {zd['mid_n']:,} | ~50th pct"),
            (top_rgb, f"n = {zd['top15_n']:,} | ≥85th pct"),
        ]
        for c_idx, (im_u8, badge) in enumerate(portraits):
            ax = fig.add_subplot(top_gs[r_idx, c_idx])
            ax.imshow(_studio_portrait(im_u8, mask_384), interpolation="bicubic")
            _style_face_ax(ax, title_top=col_titles[c_idx] if r_idx == 0 else None, badge_bot=badge)
            if c_idx == 0:
                ax.text(
                    -0.12,
                    0.5,
                    f"{g_label.split()[-1]}\n(N={zd['eval_n']:,})",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=7.6,
                    color=g_color,
                )
                ax.text(
                    0.0,
                    1.34 if r_idx == 0 else 1.10,
                    g_label if r_idx == 0 else "B   Men",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontweight="bold",
                    fontsize=8.2,
                    color=INK,
                )

        # Col 3: Rejector map on Bot15/Mid relief
        ax_rej = fig.add_subplot(top_gs[r_idx, 3])
        im_rej, m_rej = _studio_heatmap(bot_rgb, npz[f"map__tier__{g}__bot15"], mask_384, vmax_tier)
        ax_rej.imshow(im_rej, interpolation="bicubic")
        ax_rej.contour(m_rej, levels=[-0.52 * vmax_tier], colors=["#08306B"], linewidths=0.55, linestyles="--")
        _style_face_ax(ax_rej, title_top=col_titles[3] if r_idx == 0 else None, badge_bot="−Δ pct foci", border_color="#9BB8D8")

        # Col 4: Driver map on Top15/Mid relief
        ax_drv = fig.add_subplot(top_gs[r_idx, 4])
        im_drv, m_drv = _studio_heatmap(top_rgb, npz[f"map__tier__{g}__top15"], mask_384, vmax_tier)
        ax_drv.imshow(im_drv, interpolation="bicubic")
        ax_drv.contour(m_drv, levels=[0.52 * vmax_tier], colors=["#7F1D1D"], linewidths=0.55)
        _style_face_ax(ax_drv, title_top=col_titles[4] if r_idx == 0 else None, badge_bot="+Δ pct foci", border_color="#E5B299")

        # Col 5: 218-triangle Delaunay geometry map + wireframe
        ax_geo = fig.add_subplot(top_gs[r_idx, 5])
        im_geo, _ = _studio_heatmap(mid_rgb, npz[f"map__geom_diff__{g}"], mask_384, vmax_geom)
        ax_geo.imshow(im_geo, interpolation="bicubic")
        ax_geo.triplot(canon_pts[:, 0], canon_pts[:, 1], tris, color="#1E293B", lw=0.26, alpha=0.34)
        _style_face_ax(ax_geo, title_top=col_titles[5] if r_idx == 0 else None, badge_bot="106-pt mesh", border_color="#C8C7C0")

    # Centered horizontal diverging colorbar
    cax = fig.add_subplot(cbar_gs[0, 1])
    sm = plt.cm.ScalarMappable(cmap=FACE_CMAP, norm=plt.Normalize(vmin=-vmax_tier, vmax=vmax_tier))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(MUTED)
    cb.ax.tick_params(labelsize=6.3, length=2.0, pad=1.5)
    cb.set_label("Local beauty rank attribution (Δ within-gender percentile points: sapphire = rejector, coral-gold = driver)", fontsize=6.6, labelpad=2.0, color=INK2)

    # Panel C: 12-Zone Quantitative Beauty Drivers (+Δ pct) & Rejectors (−Δ pct)
    ax_c = fig.add_subplot(bot_gs[0, 0])
    panel(ax_c, "C", "12-zone beauty drivers (+Δ) & rejectors (−Δ)", xlabel="Zone contribution to within-gender beauty percentile (95% CI)", fontsize=7.9)
    ax_c.grid(False)
    ax_c.grid(axis="x", color=GRID, linewidth=0.55)
    ax_c.axvline(0, color=INK, lw=0.75)

    w_zones = {z["zone_key"]: z for z in fa["zone_drivers_and_rejectors"]["Female"]["zones"]}
    m_zones = {z["zone_key"]: z for z in fa["zone_drivers_and_rejectors"]["Male"]["zones"]}
    ordered_keys = sorted(
        w_zones.keys(),
        key=lambda k: 0.5 * (w_zones[k]["spread_top_minus_bot_pct"] + m_zones[k]["spread_top_minus_bot_pct"]),
    )
    y_pos = np.arange(len(ordered_keys))
    bh = 0.34
    for idx_k, zk in enumerate(ordered_keys):
        wz, mz = w_zones[zk], m_zones[zk]
        # Top-15% positive drivers (right of 0)
        ax_c.barh(idx_k + bh / 2, wz["top15_driver_pct"], height=bh, color=WOMEN, alpha=0.90, zorder=3)
        ax_c.barh(idx_k - bh / 2, mz["top15_driver_pct"], height=bh, color=MEN, alpha=0.90, zorder=3)
        ax_c.plot(wz["top15_ci95"], [idx_k + bh / 2] * 2, color="#7C2D12", lw=0.85, zorder=4)
        ax_c.plot(mz["top15_ci95"], [idx_k - bh / 2] * 2, color="#0C4A6E", lw=0.85, zorder=4)
        # Bottom-15% negative rejectors (left of 0)
        ax_c.barh(idx_k + bh / 2, wz["bot15_rejector_pct"], height=bh, color=WOMEN, alpha=0.45, hatch="///", edgecolor=WOMEN, lw=0.4, zorder=3)
        ax_c.barh(idx_k - bh / 2, mz["bot15_rejector_pct"], height=bh, color=MEN, alpha=0.45, hatch="///", edgecolor=MEN, lw=0.4, zorder=3)
        ax_c.plot(wz["bot15_ci95"], [idx_k + bh / 2] * 2, color="#7C2D12", lw=0.85, zorder=4)
        ax_c.plot(mz["bot15_ci95"], [idx_k - bh / 2] * 2, color="#0C4A6E", lw=0.85, zorder=4)

    ax_c.set_yticks(y_pos, [_short_zone_label(k) for k in ordered_keys], fontsize=6.5)
    ax_c.set_ylim(-0.65, len(ordered_keys) - 0.35)
    ax_c.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc=WOMEN, alpha=0.9, label="Women driver / rejector"),
            plt.Rectangle((0, 0), 1, 1, fc=MEN, alpha=0.9, label="Men driver / rejector"),
        ],
        loc="lower left",
        fontsize=6.2,
    )

    # Panel D: Single-channel visual ceiling (H1 -> 106-pt -> HoloFace-Pure -> HoloFace-Full -> Oracle)
    ax_d = fig.add_subplot(bot_gs[0, 1])
    panel(ax_d, "D", "Visual anatomy vs. single-context ceiling", xlabel="Share of human-jury rank variance (ρ²)", fontsize=7.9)
    ax_d.grid(False)
    ax_d.grid(axis="x", color=GRID, linewidth=0.55)

    bm_waves = {w["wave_id"]: w for w in data["inventory"]["multimodal_benchmark"]["waves"]}
    hb = data["holoface_benchmark"]
    r2_h1 = bm_waves["H1"]["profile_test"]["spearman_r"] ** 2
    r2_pf = fa["benchmark_10k_pure_face"]["test_overall"]["spearman_r"] ** 2
    r2_hpure = hb["holoface_pure_zero_metadata"]["test"]["spearman_r2"]
    r2_hfull = hb["holoface_full_multimodal"]["test_l1"]["spearman_r2"]
    r2_oracle = hb["psychometric_voter_ceiling"]["tiers"]["all_test"]["oracle_finite_vote_spearman_r2"]

    rungs = [
        ("Raw UV crop\n(H1 ResNet-18)", r2_h1, GREY),
        ("106-pt Delaunay atlas\n(3 backbones, zero meta)", r2_pf, AQUA),
        ("HoloFace-Pure\n(12-space + 3D + skin)", r2_hpure, B300),
        ("HoloFace-Full\n(multimodal exact-L1)", r2_hfull, MEN),
        ("Finite-voter Oracle\n(single-context bound)", r2_oracle, WOMEN),
    ]
    ry = np.arange(len(rungs))[::-1]
    for yi, (lbl, val, col) in zip(ry, rungs):
        ax_d.barh(yi, 100.0 * val, height=0.54, color=col, alpha=0.88, zorder=3)
        ax_d.text(100.0 * val + 1.4, yi, f"{100.0 * val:.1f}%", va="center", ha="left", fontsize=6.4, fontweight="bold", color=INK)
    ax_d.set_yticks(ry, [r[0] for r in rungs], fontsize=6.0)
    ax_d.set_xlim(0, 102)
    ax_d.set_ylim(-0.55, len(rungs) - 0.45)

    return pdf(fig, dpi=360)


def face_age_plot(data):
    fa = data["facial_attribution"]
    npz = _load_face_npz(data)
    mask_384 = _studio_mask_384(npz)
    age_buckets = ["18-24", "25-34", "35-44", "45-54", "55+"]

    fig = plt.figure(figsize=(7.35, 6.10))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.34, 1.18], wspace=0.42)
    left_outer = outer[0].subgridspec(2, 1, hspace=0.30)
    sex_grids = {
        "Female": left_outer[0].subgridspec(2, 5, wspace=0.06, hspace=0.14),
        "Male": left_outer[1].subgridspec(2, 5, wspace=0.06, hspace=0.14),
    }
    right_gs = outer[1].subgridspec(2, 1, hspace=0.42)

    age_vals = []
    for g in ("Female", "Male"):
        for ab in age_buckets:
            s_ab = ab.replace("+", "plus")
            age_vals.append(np.percentile(np.abs(npz[f"map__age__{g}__{s_ab}"].astype(np.float32)), 98.5))
    vmax_age = max(float(max(age_vals)), 1.2)

    row_specs = [
        ("Female", 0, "comp", "A   Women: super-resolved composites & signed attribution across age", WOMEN),
        ("Female", 1, "map", None, WOMEN),
        ("Male", 0, "comp", "Men: super-resolved composites & signed attribution across age", MEN),
        ("Male", 1, "map", None, MEN),
    ]
    def _ord(v):
        k = int(round(v))
        suf = "th" if 11 <= (k % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
        return f"{k}{suf}"

    for g, sub_r, kind, sec_title, g_col in row_specs:
        for c_i, ab in enumerate(age_buckets):
            ax = fig.add_subplot(sex_grids[g][sub_r, c_i])
            s_ab = ab.replace("+", "plus")
            cell = fa["age_bucket_progression"][g][ab]
            comp_u8 = npz[f"comp__age__{g}__{s_ab}"]
            if kind == "comp":
                ax.imshow(_studio_portrait(comp_u8, mask_384), interpolation="bicubic")
                badge = f"n={cell['n']} | {_ord(cell['mean_observed_pct'])}"
                _style_face_ax(ax, title_top=f"Ages {ab}", badge_bot=badge)
            else:
                im_h, _ = _studio_heatmap(comp_u8, npz[f"map__age__{g}__{s_ab}"], mask_384, vmax_age)
                ax.imshow(im_h, interpolation="bicubic")
                badge = f"face {_ord(cell['mean_predicted_pct'])}"
                _style_face_ax(ax, title_top=None, badge_bot=badge, border_color="#BFCAD6")
            if c_i == 0:
                row_tag = "Portrait" if kind == "comp" else "ΔPct map"
                ax.text(-0.11, 0.5, row_tag, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.3, fontweight="bold", color=g_col)
                if sec_title:
                    ax.text(0.0, 1.30, sec_title, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=7.9, color=INK)

    # Panel B (top right): 55+ minus 18-24 Decomposition (Voter Drop = Facial Anatomy + Non-Facial Age Penalty)
    ax_b = fig.add_subplot(right_gs[0, 0])
    panel(ax_b, "B", "Aging drop: anatomy vs. label penalty", xlabel="55+ minus 18\u201324 rank shift (pct. pts)", fontsize=7.8)
    ax_b.grid(False)
    ax_b.grid(axis="x", color=GRID, linewidth=0.55)
    ax_b.axvline(0.0, color=INK, lw=0.7)
    w_obs_d = fa["age_bucket_progression"]["Female"]["55+"]["mean_observed_pct"] - fa["age_bucket_progression"]["Female"]["18-24"]["mean_observed_pct"]
    w_prd_d = fa["age_bucket_progression"]["Female"]["55+"]["mean_predicted_pct"] - fa["age_bucket_progression"]["Female"]["18-24"]["mean_predicted_pct"]
    w_pen_d = w_obs_d - w_prd_d
    m_obs_d = fa["age_bucket_progression"]["Male"]["55+"]["mean_observed_pct"] - fa["age_bucket_progression"]["Male"]["18-24"]["mean_observed_pct"]
    m_prd_d = fa["age_bucket_progression"]["Male"]["55+"]["mean_predicted_pct"] - fa["age_bucket_progression"]["Male"]["18-24"]["mean_predicted_pct"]
    m_pen_d = m_obs_d - m_prd_d
    decomp_rows = [
        ("Observed voter\nrank shift", w_obs_d, m_obs_d),
        ("Pure facial\nanatomy shift", w_prd_d, m_prd_d),
        ("Non-facial age\nlabel penalty", w_pen_d, m_pen_d),
    ]
    yb_d = np.arange(len(decomp_rows))
    for i, (_, wv, mv) in enumerate(decomp_rows):
        ax_b.barh(i + 0.17, wv, height=0.31, color=WOMEN, alpha=0.90, label="Women" if i == 0 else None)
        ax_b.barh(i - 0.17, mv, height=0.31, color=MEN, alpha=0.90, label="Men" if i == 0 else None)
        ax_b.text(wv - 1.1 if wv < 0 else wv + 1.1, i + 0.17, f"{wv:+.1f}", va="center", ha="right" if wv < 0 else "left", fontsize=5.2, fontweight="bold", color=INK)
        ax_b.text(mv - 1.1 if mv < 0 else mv + 1.1, i - 0.17, f"{mv:+.1f}", va="center", ha="right" if mv < 0 else "left", fontsize=5.2, fontweight="bold", color=INK)
    ax_b.set_yticks(yb_d, [r[0] for r in decomp_rows], fontsize=6.0)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-52.0, 12.0)
    ax_b.legend(loc="lower left", fontsize=5.6)

    # Panel C (bottom right): Anatomical Zone Shift from 18-24 to 55+ (55+ minus 18-24)
    ax_c = fig.add_subplot(right_gs[1, 0])
    panel(ax_c, "C", "Zone aging drift (55+ minus 18–24)", xlabel="Δ percentile contribution", fontsize=7.8)
    ax_c.grid(False)
    ax_c.grid(axis="x", color=GRID, linewidth=0.55)
    ax_c.axvline(0.0, color=INK, lw=0.7)
    focus_zones = ["tear_trough", "nasolabial", "jaw_chin", "lips_vermilion", "eyes_iris", "malar_cheek"]
    zy = np.arange(len(focus_zones))
    for idx_z, zk in enumerate(focus_zones):
        dw = fa["age_bucket_progression"]["Female"]["55+"]["zone_deltas_pct"][zk] - fa["age_bucket_progression"]["Female"]["18-24"]["zone_deltas_pct"][zk]
        dm = fa["age_bucket_progression"]["Male"]["55+"]["zone_deltas_pct"][zk] - fa["age_bucket_progression"]["Male"]["18-24"]["zone_deltas_pct"][zk]
        ax_c.barh(idx_z + 0.17, dw, height=0.32, color=WOMEN, alpha=0.90)
        ax_c.barh(idx_z - 0.17, dm, height=0.32, color=MEN, alpha=0.90)
    ax_c.set_yticks(zy, [_short_zone_label(z) for z in focus_zones], fontsize=6.3)
    ax_c.set_ylim(-0.6, len(focus_zones) - 0.4)

    return pdf(fig, dpi=360)


def face_world_plot(data):
    fa = data["facial_attribution"]
    npz = _load_face_npz(data)
    mask_384 = _studio_mask_384(npz)

    regions = [
        ("United States", "United\nStates", "US"),
        ("South America", "South\nAmerica", "S.Am"),
        ("Mexico, Central Am. & Caribbean", "Mexico &\nCaribbean", "MCAC"),
        ("Canada", "Canada", "CAN"),
        ("Asia-Pacific & Oceania", "Asia-Pacific\n& Oceania", "APAC"),
        ("Europe", "Europe", "EUR"),
        ("Middle East & Africa", "Middle East\n& Africa", "MEA"),
    ]
    # Keep regions present in both Female and Male syntheses
    avail_regs = [
        r for r in regions
        if r[0] in fa["world_regional_synthesis"]["Female"]["regions"]
        and r[0] in fa["world_regional_synthesis"]["Male"]["regions"]
    ]
    n_reg = len(avail_regs)

    fig = plt.figure(figsize=(7.35, 7.15))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.46, 1.08], hspace=0.33)
    top_outer = outer[0].subgridspec(2, 1, hspace=0.22)
    reg_sex_gs = {
        "Female": top_outer[0].subgridspec(2, n_reg, wspace=0.055, hspace=0.13),
        "Male": top_outer[1].subgridspec(2, n_reg, wspace=0.055, hspace=0.13),
    }
    bot_gs = outer[1].subgridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.36)

    drv_vals = []
    for g in ("Female", "Male"):
        for r_full, _, _ in avail_regs:
            s_reg = r_full.replace(",", "").replace(" ", "_").replace(".", "").replace("&", "and")
            drv_vals.append(np.percentile(np.abs(npz[f"map__region_driver__{g}__{s_reg}"].astype(np.float32)), 98.5))
    vmax_reg = max(float(max(drv_vals)), 1.2)

    row_cfg = [
        ("Female", 0, "comp", "A   Global regional super-resolved composites & within-region beauty driver maps (Women & Men)", WOMEN),
        ("Female", 1, "drv", None, WOMEN),
        ("Male", 0, "comp", None, MEN),
        ("Male", 1, "drv", None, MEN),
    ]
    for g, sub_r, mode, banner, g_col in row_cfg:
        for c_i, (r_full, r_title, _) in enumerate(avail_regs):
            ax = fig.add_subplot(reg_sex_gs[g][sub_r, c_i])
            s_reg = r_full.replace(",", "").replace(" ", "_").replace(".", "").replace("&", "and")
            rcell = fa["world_regional_synthesis"][g]["regions"][r_full]
            comp_u8 = npz[f"comp__region__{g}__{s_reg}"]
            if mode == "comp":
                ax.imshow(_studio_portrait(comp_u8, mask_384), interpolation="bicubic")
                _style_face_ax(ax, title_top=r_title if (g == "Female" and sub_r == 0) else None, badge_bot=f"n = {rcell['n']:,}")
            else:
                im_d, _ = _studio_heatmap(comp_u8, npz[f"map__region_driver__{g}__{s_reg}"], mask_384, vmax_reg)
                ax.imshow(im_d, interpolation="bicubic")
                _style_face_ax(ax, title_top=None, badge_bot="Hi − Lo Δpct", border_color="#BFCAD6")
            if c_i == 0:
                sex_tag = "Women" if g == "Female" else "Men"
                sub_tag = f"{sex_tag}\nface" if mode == "comp" else f"{sex_tag}\ndrivers"
                ax.text(-0.14, 0.5, sub_tag, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.0, fontweight="bold", color=g_col)
                if banner:
                    ax.text(0.0, 1.40, banner, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=7.9, color=INK)

    # Panel B: Cross-regional spatial driver consistency matrix (lower triangle = Women, upper triangle = Men)
    ax_b = fig.add_subplot(bot_gs[0, 0])
    panel(ax_b, "B", "Cross-regional driver correlation (r)", fontsize=7.8)
    ax_b.grid(False)
    mat = np.eye(n_reg, dtype=np.float32)
    cw = fa["world_regional_synthesis"]["Female"]["cross_region_driver_corr"]
    cm = fa["world_regional_synthesis"]["Male"]["cross_region_driver_corr"]
    for i in range(n_reg):
        for j in range(n_reg):
            ri, rj = avail_regs[i][0], avail_regs[j][0]
            if i > j:
                mat[i, j] = cw[ri][rj]
            elif i < j:
                mat[i, j] = cm[ri][rj]
    im_m = ax_b.imshow(mat, vmin=0.35, vmax=1.0, cmap="YlGnBu", aspect="auto")
    short_codes = [r[2] for r in avail_regs]
    ax_b.set_xticks(range(n_reg), short_codes, fontsize=6.1, rotation=28, ha="right")
    ax_b.set_yticks(range(n_reg), short_codes, fontsize=6.1)
    for i in range(n_reg):
        for j in range(n_reg):
            val = mat[i, j]
            txt = "—" if i == j else f"{val:.2f}"
            col_t = "white" if (i != j and val >= 0.75) else INK
            ax_b.text(j, i, txt, ha="center", va="center", fontsize=5.5, color=col_t, fontweight="bold" if i != j else "normal")
    ax_b.set_xlabel("Lower tri.: Women  |  Upper tri.: Men", fontsize=6.3, labelpad=2.5)

    # Panel C: Regional observed vs. face-predicted mean beauty rank across all 7 regions
    ax_c = fig.add_subplot(bot_gs[0, 1])
    panel(ax_c, "C", "Regional consensus vs. pure-face prediction", xlabel="World region", ylabel="Mean beauty percentile", fontsize=7.8)
    rx = np.arange(n_reg)
    ax_c.axhline(50.0, color=MUTED, ls=":", lw=0.7)
    all_reg_y = []
    for (g, lbl, col, mk), off in zip(SERIES, (-0.13, 0.13)):
        obs_r = [fa["world_regional_synthesis"][g]["regions"][r[0]]["mean_observed_pct"] for r in avail_regs]
        prd_r = [fa["world_regional_synthesis"][g]["regions"][r[0]]["mean_predicted_pct"] for r in avail_regs]
        all_reg_y.extend(obs_r + prd_r)
        ax_c.scatter(rx + off, obs_r, color=col, marker=mk, s=22, zorder=4, label=f"{lbl} voter rank")
        ax_c.scatter(rx + off, prd_r, facecolor="white", edgecolor=col, marker=mk, s=20, lw=1.1, zorder=4, label=f"{lbl} face-only")
        for xi, yo, yp in zip(rx + off, obs_r, prd_r):
            ax_c.plot([xi, xi], [yo, yp], color=col, lw=0.75, alpha=0.65, zorder=3)
    ax_c.set_xticks(rx, short_codes, fontsize=6.3)
    ax_c.set_ylim(max(0.0, min(all_reg_y) - 14.5), min(100.0, max(all_reg_y) + 7.0))
    ax_c.legend(loc="lower left", fontsize=5.8, ncol=2, frameon=True, facecolor="white", framealpha=0.90, edgecolor="none")

    return pdf(fig, dpi=360)


def face_drivers_table(data):
    fa = data["facial_attribution"]
    wz = {z["zone_key"]: z for z in fa["zone_drivers_and_rejectors"]["Female"]["zones"]}
    mz = {z["zone_key"]: z for z in fa["zone_drivers_and_rejectors"]["Male"]["zones"]}
    ordered = sorted(
        wz.keys(),
        key=lambda k: 0.5 * (wz[k]["spread_top_minus_bot_pct"] + mz[k]["spread_top_minus_bot_pct"]),
        reverse=True,
    )
    lines = [
        r"& \multicolumn{3}{c}{Women ($N=" + f"{fa['zone_drivers_and_rejectors']['Female']['eval_n']:,}" + r"$)} & \multicolumn{3}{c}{Men ($N=" + f"{fa['zone_drivers_and_rejectors']['Male']['eval_n']:,}" + r"$)}\\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Canonical anatomical zone (106-pt Delaunay atlas) & Top 15\% driver & Bot.\ 15\% rejector & Spread & Top 15\% driver & Bot.\ 15\% rejector & Spread\\\midrule",
    ]
    for zk in ordered:
        w, m = wz[zk], mz[zk]
        lbl = w["zone_label"].replace("&", r"\&")
        w_d = f"${w['top15_driver_pct']:+.2f}$ $[{w['top15_ci95'][0]:+.2f},{w['top15_ci95'][1]:+.2f}]$"
        w_r = f"${w['bot15_rejector_pct']:+.2f}$ $[{w['bot15_ci95'][0]:+.2f},{w['bot15_ci95'][1]:+.2f}]$"
        m_d = f"${m['top15_driver_pct']:+.2f}$ $[{m['top15_ci95'][0]:+.2f},{m['top15_ci95'][1]:+.2f}]$"
        m_r = f"${m['bot15_rejector_pct']:+.2f}$ $[{m['bot15_ci95'][0]:+.2f},{m['bot15_ci95'][1]:+.2f}]$"
        lines.append(
            f"{lbl} & {w_d} & {w_r} & ${w['spread_top_minus_bot_pct']:.2f}$ & "
            f"{m_d} & {m_r} & ${m['spread_top_minus_bot_pct']:.2f}$\\\\"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 9. NUMBERS MACROS & GENERATED TABLES
# ---------------------------------------------------------------------------
def format_rate(x):
    return f"{100 * x:.1f}\\%"


def format_pp(x):
    return f"${x:+.1f}$"


def interval(cell, scale=1, key="interval_95", digits=1):
    lo, hi = (scale * v for v in cell[key])
    return f"$[{lo:.{digits}f}, {hi:.{digits}f}]$"


def numbers(data):
    pov, hon, speed, osyn = data["pov"], data["hon"], data["speed_dating"], data["open_synthesis"]
    fa = data["facial_attribution"]
    main = hon["ratings"]["votes_20"]["all_ages"]
    att = hon["attention_and_portfolio"]
    cn = hon["cross_national"]
    hp_m = pov["height_percentiles"]["by_sex"]["Male"]
    cliff = hp_m["cliff_5ft11_6ft0_6ft1"]
    whyte = osyn["whyte_2021"]
    lid = osyn["lidborg_2022"]
    kow = osyn["kowal_2025"]
    tw = osyn["two_way_jury"]
    bm = data["inventory"]["multimodal_benchmark"]["waves"]

    m_br = {b["height_code"]: b for b in whyte["height_vs_attraction"]["Men"]["by_bracket"]}
    m_mv_a = whyte["height_vs_attraction"]["Men"]["multivariate_standardized"]["attraction_pct_per_sd"]
    m_mv_p = whyte["height_vs_attraction"]["Men"]["multivariate_standardized"]["partnered_pp_per_sd"]
    m_quad = whyte["height_vs_attraction"]["Men"]["quadratic_fit_pct_per_sd"]["quadratic"]

    inch_map = {r["inches"]: r for r in hp_m["by_inch"]}

    values = {
        # Existing core macros
        "PboPovRows": f"{pov['accounts']['rows']:,}",
        "PboPovEligible": f"{pov['eligible']['opposite_gender']:,}",
        "PboHonRows": f"{hon['inventory']['canonical_profiles']:,}",
        "PboHonEligible": f"{sum(g['all']['n'] for g in main.values()):,}",
        "PboHonImages": f"{sum(g['inventoried']['n'] for g in main.values()):,}",
        "PboHonFiles": f"{hon['inventory']['inventoried_images']:,}",
        "PboBeautyN": f"{sum(x['reference_n'] for x in hon['beauty_reference'].values()):,}",
        "PboSpeedN": f"{speed['counts']['decisions_with_looks_panel']:,}",
        "PboSpeedSessions": str(speed["sessions"]),
        "PboMinCell": str(data["cohorts"]["min_profiles_per_cell"]),
        "PboCombinedRecords": f"{pov['accounts']['rows'] + hon['inventory']['canonical_profiles']:,}",

        # HON attention, portfolio & cross-national macros
        "PboHonVotesMedW": f"{att['by_sex']['Female']['median_votes']:,.0f}",
        "PboHonVotesMedM": f"{att['by_sex']['Male']['median_votes']:,.0f}",
        "PboHonAttentionRatio": f"{att['by_sex']['Female']['median_votes'] / att['by_sex']['Male']['median_votes']:.1f}",
        "PboHonVotesGiniW": f"{att['by_sex']['Female']['votes_gini']:.3f}",
        "PboHonVotesGiniM": f"{att['by_sex']['Male']['votes_gini']:.3f}",
        "PboHonLikesGiniW": f"{att['by_sex']['Female']['likes_gini']:.3f}",
        "PboHonLikesGiniM": f"{att['by_sex']['Male']['likes_gini']:.3f}",
        "PboHonTopDecileLikesW": format_rate(att["by_sex"]["Female"]["top_10pct_likes_share"]),
        "PboHonTopDecileLikesM": format_rate(att["by_sex"]["Male"]["top_10pct_likes_share"]),
        "PboHonCountriesHundred": str(cn["counts_by_threshold"]["ge_100_per_sex"]),
        "PboHonCountriesFiveHundred": str(cn["counts_by_threshold"]["ge_500_per_sex"]),
        "PboHonCountriesStrictAge": str(cn["summary_strict_age_contrast"]["countries"]),
        "PboHonFemaleSteeperCountries": str(cn["summary_strict_age_contrast"]["female_steeper_drop_countries"]),
        "PboHonAgeDropMedW": f"{cn['summary_strict_age_contrast']['median_female_drop_pp']:.1f}",
        "PboHonAgeDropMedM": f"{cn['summary_strict_age_contrast']['median_male_drop_pp']:.1f}",
        "PboHonPartLikeCorr": f"{kow['kowal_hon_bridge']['hon_500_log_participation_vs_like_gap_r']:.2f}",

        # POV commitment horizon, union durability & postponement macros
        "PboPovSeriousYoungW": format_rate(pov["market_and_commitment"]["25-29"]["by_sex"]["Female"]["serious_commitment"]["share"]),
        "PboPovSeriousYoungM": format_rate(pov["market_and_commitment"]["25-29"]["by_sex"]["Male"]["serious_commitment"]["share"]),
        "PboPovSeriousMidW": format_rate(pov["market_and_commitment"]["40-49"]["by_sex"]["Female"]["serious_commitment"]["share"]),
        "PboPovSeriousMidM": format_rate(pov["market_and_commitment"]["40-49"]["by_sex"]["Male"]["serious_commitment"]["share"]),
        "PboPovSeriousSenW": format_rate(pov["market_and_commitment"]["60-69"]["by_sex"]["Female"]["serious_commitment"]["share"]),
        "PboPovSeriousSenM": format_rate(pov["market_and_commitment"]["60-69"]["by_sex"]["Male"]["serious_commitment"]["share"]),
        "PboPovTenYrMidW": format_rate(pov["market_and_commitment"]["40-49"]["by_sex"]["Female"]["over_10y_union"]["share"]),
        "PboPovTenYrMidM": format_rate(pov["market_and_commitment"]["40-49"]["by_sex"]["Male"]["over_10y_union"]["share"]),
        "PboPovAmbParentLowW": format_rate(pov["traits_and_postponement"]["ambition_ages_25_49"]["Not Ambitious"]["Female"]["parenthood"]["share"]),
        "PboPovAmbParentHighW": format_rate(pov["traits_and_postponement"]["ambition_ages_25_49"]["Very Ambitious"]["Female"]["parenthood"]["share"]),
        "PboPovAmbWantLowW": format_rate(pov["traits_and_postponement"]["ambition_ages_25_49"]["Not Ambitious"]["Female"]["want_children_all"]["share"]),
        "PboPovAmbWantHighW": format_rate(pov["traits_and_postponement"]["ambition_ages_25_49"]["Very Ambitious"]["Female"]["want_children_all"]["share"]),

        # Male Height Percentile vs Attraction, 6'0" Staging & Meta-Analysis macros
        "PboHeightMenN": f"{hp_m['n_valid']:,}",
        "PboHeightFiveElevenN": f"{cliff['n_5ft11']:,}",
        "PboHeightSixZeroN": f"{cliff['n_6ft0']:,}",
        "PboHeightSixOneN": f"{cliff['n_6ft1']:,}",
        "PboHeightCliffDrop": f"{abs(cliff['drop_6ft0_to_6ft1_pct']):.1f}\\%",
        "PboHeightSixZeroRatio": f"{inch_map[72]['observed_to_gaussian_ratio']:.2f}",
        "PboHeightUnionShort": format_rate(inch_map[61]["outcomes_ages_30_49"]["over_3y_union"]["share"]),
        "PboHeightUnionPeak": format_rate(inch_map[73]["outcomes_ages_30_49"]["over_3y_union"]["share"]),
        "PboHeightUnionTall": format_rate(inch_map[78]["outcomes_ages_30_49"]["over_3y_union"]["share"]),
        "PboHeightCasualShort": format_rate(inch_map[61]["outcomes_ages_30_49"]["casual_intent"]["share"]),
        "PboHeightCasualTall": format_rate(inch_map[76]["outcomes_ages_30_49"]["casual_intent"]["share"]),
        "PboWhyteN": f"{whyte['records']:,}",
        "PboWhyteMenN": f"{whyte['men']:,}",
        "PboWhyteWomenN": f"{whyte['women']:,}",
        "PboWhyteShortAttrPct": f"{m_br[2]['mean_attraction_percentile']:.1f}",
        "PboWhytePeakHeightPct": f"{m_br[5]['height_percentile']:.1f}",
        "PboWhytePeakAttrPct": f"{m_br[5]['mean_attraction_percentile']:.1f}",
        "PboWhyteTallHeightPct": f"{m_br[6]['height_percentile']:.1f}",
        "PboWhyteTallAttrPct": f"{m_br[6]['mean_attraction_percentile']:.1f}",
        "PboWhyteQuadBeta": f"{m_quad['estimate']:.2f}",
        "PboWhyteQuadCI": interval(m_quad, 1, "ci_95", 2),
        "PboWhyteHealthSlope": f"{m_mv_a['health']['estimate']:.1f}",
        "PboWhyteHeightSlope": f"{m_mv_a['height']['estimate']:.2f}",
        "PboWhyteIncomePartSlope": f"{m_mv_p['income']['estimate']:.1f}",
        "PboWhyteHeightPartSlope": f"{m_mv_p['height']['estimate']:.2f}",
        "PboLidborgHeightMatK": str(lid["height"]["mating"]["k"]),
        "PboLidborgHeightMatN": f"{lid['height']['mating']['total_n']:,}",
        "PboLidborgHeightMatR": f"{lid['height']['mating']['re_mean_r']:.3f}",
        "PboLidborgHeightMatCI": interval(lid["height"]["mating"], 1, "re_ci_95", 3),
        "PboLidborgHeightRepK": str(lid["height"]["reproduction_all"]["k"]),
        "PboLidborgHeightRepN": f"{lid['height']['reproduction_all']['total_n']:,}",
        "PboLidborgHeightRepR": f"{lid['height']['reproduction_all']['re_mean_r']:.3f}",
        "PboLidborgHeightRepCI": interval(lid["height"]["reproduction_all"], 1, "re_ci_95", 3),
        "PboLidborgBodyRepR": f"{lid['body_strength']['reproduction_all']['re_mean_r']:.3f}",
        "PboLidborgBodyRepCI": interval(lid["body_strength"]["reproduction_all"], 1, "re_ci_95", 3),

        # Whyte Trust, Kowal 53-country, Two-Way Jury & Benchmark macros
        "PboWhyteTrustW": f"{whyte['stated_importance']['pooled']['Women']['trust']['mean']:.1f}",
        "PboWhyteTrustM": f"{whyte['stated_importance']['pooled']['Men']['trust']['mean']:.1f}",
        "PboWhyteIntelW": f"{whyte['stated_importance']['pooled']['Women']['intelligence']['mean']:.1f}",
        "PboWhyteIntelM": f"{whyte['stated_importance']['pooled']['Men']['intelligence']['mean']:.1f}",
        "PboKowalBudgetN": f"{kow['inclusion_flow']['sum_equals_30']:,}",
        "PboKowalCountryN": str(kow["cross_country_53"]["countries"]),
        "PboKowalStdCountryN": str(kow["age_standardized_32"]["countries"]),
        "PboKowalRelCorr": f"{kow['cross_country_53']['by_item']['religion']['cross_country_men_women_r']:.2f}",
        "PboKowalKindCorr": f"{kow['cross_country_53']['by_item']['kindness']['cross_country_men_women_r']:.2f}",
        "PboKowalBridgeN": str(kow["kowal_hon_bridge"]["overlapping_countries"]),
        "PboTwoWaySingleAsym": f"{tw['single_context_asymptote_r2']:.2f}",
        "PboTwoWayEightEight": f"{tw['surface_r2']['8']['8']:.2f}",
        "PboBenchBaselineR": f"{bm[0]['profile_test']['spearman_r']:.3f}",
        "PboBenchFinalR": f"{max(w['profile_test']['spearman_r'] for w in bm):.3f}",
        "PboBenchUnexplainedPct": f"{100 * (1 - max(w['profile_test']['spearman_r'] for w in bm) ** 2):.1f}\\%",

        # Deep-learning facial attribution, super-resolved composites & regional invariance macros
        "PboFaceTotalN": f"{fa['total_aligned_portraits_processed']:,}",
        "PboFaceTrisN": str(fa["delaunay_triangles"]),
        "PboFaceEvalNatW": f"{fa['zone_drivers_and_rejectors']['Female']['eval_n']:,}",
        "PboFaceEvalNatM": f"{fa['zone_drivers_and_rejectors']['Male']['eval_n']:,}",
        "PboFacePureSpearman": f"{fa['benchmark_10k_pure_face']['test_overall']['spearman_r']:.3f}",
        "PboFacePureRankVar": f"{100 * fa['benchmark_10k_pure_face']['test_overall']['spearman_r'] ** 2:.1f}\\%",
        "PboFacePurePctMae": f"{fa['benchmark_10k_pure_face']['test_overall']['gender_pct_mae']:.1f}",
        "PboFaceLiveSpearmanW": f"{fa['live_spatial_model_verification']['Female']['test_spearman_r']:.3f}",
        "PboFaceLiveSpearmanM": f"{fa['live_spatial_model_verification']['Male']['test_spearman_r']:.3f}",
        "PboFaceTopZoneNameW": fa["zone_drivers_and_rejectors"]["Female"]["zones"][0]["zone_label"].replace("&", r"\&").lower(),
        "PboFaceTopZoneSpreadW": f"{fa['zone_drivers_and_rejectors']['Female']['zones'][0]['spread_top_minus_bot_pct']:.2f}",
        "PboFaceTopZoneDrvW": f"{fa['zone_drivers_and_rejectors']['Female']['zones'][0]['top15_driver_pct']:+.2f}",
        "PboFaceTopZoneRejW": f"{fa['zone_drivers_and_rejectors']['Female']['zones'][0]['bot15_rejector_pct']:+.2f}",
        "PboFaceTopZoneNameM": fa["zone_drivers_and_rejectors"]["Male"]["zones"][0]["zone_label"].replace("&", r"\&").lower(),
        "PboFaceTopZoneSpreadM": f"{fa['zone_drivers_and_rejectors']['Male']['zones'][0]['spread_top_minus_bot_pct']:.2f}",
        "PboFaceTopZoneDrvM": f"{fa['zone_drivers_and_rejectors']['Male']['zones'][0]['top15_driver_pct']:+.2f}",
        "PboFaceTopZoneRejM": f"{fa['zone_drivers_and_rejectors']['Male']['zones'][0]['bot15_rejector_pct']:+.2f}",
        "PboFaceAgeObsYoungW": f"{fa['age_bucket_progression']['Female']['18-24']['mean_observed_pct']:.1f}",
        "PboFaceAgeObsOldW": f"{fa['age_bucket_progression']['Female']['55+']['mean_observed_pct']:.1f}",
        "PboFaceAgePredYoungW": f"{fa['age_bucket_progression']['Female']['18-24']['mean_predicted_pct']:.1f}",
        "PboFaceAgePredOldW": f"{fa['age_bucket_progression']['Female']['55+']['mean_predicted_pct']:.1f}",
        "PboFaceAgeObsYoungM": f"{fa['age_bucket_progression']['Male']['18-24']['mean_observed_pct']:.1f}",
        "PboFaceAgeObsOldM": f"{fa['age_bucket_progression']['Male']['55+']['mean_observed_pct']:.1f}",
        "PboFaceAgePredYoungM": f"{fa['age_bucket_progression']['Male']['18-24']['mean_predicted_pct']:.1f}",
        "PboFaceAgePredOldM": f"{fa['age_bucket_progression']['Male']['55+']['mean_predicted_pct']:.1f}",
        "PboFaceWorldRegionsN": str(len(fa["world_regional_synthesis"]["Female"]["regions"])),
        "PboFaceWorldCorrW": f"{fa['world_regional_synthesis']['Female']['mean_pairwise_driver_corr']:.2f}",
        "PboFaceWorldCorrM": f"{fa['world_regional_synthesis']['Male']['mean_pairwise_driver_corr']:.2f}",
        "PboFaceWorldMinCorrW": f"{fa['world_regional_synthesis']['Female']['min_pairwise_driver_corr']:.2f}",
        "PboFaceWorldMinCorrM": f"{fa['world_regional_synthesis']['Male']['min_pairwise_driver_corr']:.2f}",
    }

    # Paleoclimate & 1.5-0.7 Ma Plateau Audit macros
    pp = data["paleoclimate_plateau"]
    du_pre = pp["du2018_age_uncertainty_mc"]["slopes"]["phase_II_2.0_1.5"]["median_slope_ln_per_myr"]
    du_plt = pp["du2018_age_uncertainty_mc"]["slopes"]["phase_III_1.5_0.7"]["median_slope_ln_per_myr"]
    values.update({
        "PboPlateauMeanCC": f"{pp['plateau_summary']['plateau_level_cm3']:.0f}",
        "PboPlateauStartMa": f"{pp['plateau_summary']['published_plateau_ma'][1]:.2f}",
        "PboPlateauStartFit": f"{pp['plateau_summary']['fitted_plateau_ma'][1]:.2f}",
        "PboPlateauEndFit": f"{pp['plateau_summary']['fitted_plateau_ma'][0]:.2f}",
        "PboLinearDeltaAICc": f"{pp['plateau_summary']['linear_delta_aicc']:.1f}",
        "PboDuRatePre": f"{du_pre:.2f}",
        "PboDuRatePlateau": f"{du_plt:.2f}",
        "PboDuRateDropPct": f"{100.0 * (1.0 - du_plt / du_pre):.0f}\\%",
        "PboMptSwingPre": f"{pp['mpt_amplification']['pre_mpt_swing_permil']:.2f}",
        "PboMptSwingPost": f"{pp['mpt_amplification']['post_mpt_swing_permil']:.2f}",
        "PboMisThirtyFourD": f"{pp['mpt_amplification']['mis34_peak_permil']:.2f}",
        "PboGapMptKyr": f"{pp['timing_gap_summary']['onset_to_mpt_kyr']:.0f}",
        "PboGapCoolingKyr": f"{pp['timing_gap_summary']['onset_to_mis34_cooling_kyr']:.0f}",
        "PboGapBottleKyr": f"{pp['timing_gap_summary']['onset_to_fitcoal_window_kyr']:.0f}",
        "PboGapTektiteKyr": f"{pp['timing_gap_summary']['onset_to_bolaven_impact_kyr']:.0f}",
        "PboCcDivEqFor": f"{_v9_json('e18_cultures')['divorce_access_foragers']['share']['3']:.0f}",
    })

    # HoloFace-L1 43-Wave Benchmark, Shapley & Psychometric Voter Ceiling macros
    hb = data["holoface_benchmark"]
    hp_pure = hb["holoface_pure_zero_metadata"]
    hf_full = hb["holoface_full_multimodal"]
    pc_ceil = hb["psychometric_voter_ceiling"]
    sh_map = {r["family_id"]: r for r in hp_pure["shapley_4family_decomposition"]}
    values.update({
        "PboHoloPureSpearman": f"{hp_pure['test']['spearman_r']:.3f}",
        "PboHoloPureRankVar": format_rate(hp_pure["test"]["spearman_r2"]),
        "PboHoloPureTestMAE": f"{hp_pure['test']['beauty_mae']:.4f}",
        "PboHoloPureFiveHundredR": f"{pc_ceil['tiers']['votes_500_plus']['holoface_pure_spearman_r']:.3f}",
        "PboShapGeomShare": f"{sh_map['F1_geometry_3d']['shapley_share_of_pure_face_pct']:.1f}\\%",
        "PboShapSkinShare": f"{sh_map['F2_skin_photometry']['shapley_share_of_pure_face_pct']:.1f}\\%",
        "PboShapPoseShare": f"{sh_map['F3_staging_pose']['shapley_share_of_pure_face_pct']:.1f}\\%",
        "PboShapFoundShare": f"{sh_map['F4_foundation_12space']['shapley_share_of_pure_face_pct']:.1f}\\%",
        "PboShapAnatomyTotal": f"{100.0 - sh_map['F3_staging_pose']['shapley_share_of_pure_face_pct']:.1f}\\%",
        "PboHoloFullValMAE": f"{hf_full['val_l1']['beauty_mae']:.4f}",
        "PboHoloFullTestMAE": f"{hf_full['test_l1']['beauty_mae']:.4f}",
        "PboHoloFullSingleMAE": f"{hf_full['single_image_test']['beauty_mae']:.4f}",
        "PboHoloFullPctMAE": f"{hf_full['test_l1']['gender_pct_mae']:.1f}",
        "PboHoloFullSpearman": f"{hf_full['test_l1']['spearman_r']:.3f}",
        "PboHoloFullRankVar": format_rate(hf_full["test_l1"]["spearman_r2"]),
        "PboHoloFullFiveHundredR": f"{pc_ceil['tiers']['votes_500_plus']['holoface_full_spearman_r']:.3f}",
        "PboHoloLTwoRMSE": f"{hf_full['test_l2_unbiased_head']['beauty_rmse']:.4f}",
        "PboHoloMaleLTwoBias": f"{hf_full['subgroups_test_l2_gender']['Male']['bias']:+.3f}",
        "PboVoterSigmaSq": f"{pc_ceil['variance_components_train_fit']['pooled_train_oof_fit']['sigma_voter_var']:.2f}",
        "PboVoterSD": f"{pc_ceil['variance_components_train_fit']['pooled_train_oof_fit']['sigma_voter_sd']:.2f}",
        "PboOracleTwentyR": f"{pc_ceil['tiers']['votes_20_49']['oracle_finite_vote_spearman_r']:.3f}",
        "PboOracleFiftyR": f"{pc_ceil['tiers']['votes_50_199']['oracle_finite_vote_spearman_r']:.3f}",
        "PboOracleTwoHundredR": f"{pc_ceil['tiers']['votes_200_plus']['oracle_finite_vote_spearman_r']:.3f}",
        "PboOracleFiveHundredR": f"{pc_ceil['tiers']['votes_500_plus']['oracle_finite_vote_spearman_r']:.3f}",
        "PboOracleFiveHundredRTwo": format_rate(pc_ceil["tiers"]["votes_500_plus"]["oracle_finite_vote_spearman_r2"]),
        "PboOracleFiveHundredMAE": f"{pc_ceil['tiers']['votes_500_plus']['oracle_finite_vote_mae']:.3f}",
        "PboOracleAllR": f"{pc_ceil['tiers']['all_test']['oracle_finite_vote_spearman_r']:.3f}",
        "PboIrreducibleGapFiveHundred": format_rate(pc_ceil["tiers"]["votes_500_plus"]["irreducible_single_context_gap_full"]),
        "PboIrreducibleGapAll": format_rate(pc_ceil["tiers"]["all_test"]["irreducible_single_context_gap_full"]),
    })

    for band, age in (("25-29", "Young"), ("45-49", "Older")):
        comparison = pov["sensitivity"][band]["main"]
        values["PboPovGap" + age] = format_pp(comparison["women_minus_men_pp"])
        for sex, short in (("Female", "W"), ("Male", "M")):
            cell = comparison["responses"][sex]
            values["PboPovWant" + age + short] = format_rate(cell["shares"]["want"])
            values["PboPovOpen" + age + short] = format_rate(cell["shares"]["undecided_open"])
            values["PboPovNonresponse" + age + short] = format_rate(cell["shares"]["nonresponse"])
            beauty = hon["beauty_by_age"][band][sex]["all"]
            values["PboBeauty" + age + short] = f"{100 * beauty['mean_percentile']:.1f}"
            values["PboBeautyTop" + age + short] = format_rate(beauty["top_tenth_n"] / beauty["n"])
    for sex, label, short in (("Female", "women", "W"), ("Male", "men", "M")):
        values["PboBeautyR" + short] = f"{hon['beauty_reference'][sex]['score_like_rate_pearson_r']:.4f}"
        values["PboImageShare" + short] = format_rate(main[sex]["inventoried"]["n"] / main[sex]["all"]["n"])
        curves = speed["yes_by_panel_quintile"][label]["quintiles"]
        values["PboYesLow" + short] = format_rate(curves[0]["estimate"])
        values["PboYesHigh" + short] = format_rate(curves[-1]["estimate"])
        values["PboLooksStated" + short] = f"{speed['stated_looks'][label]['mean_looks_points_of_100']:.1f}"
        for method, name in (("own", "Own"), ("panel", "Panel")):
            coeff = speed["adjusted_slopes"][method][label]["coefficients_pp"]["attr"]
            values["PboLooks" + name + short] = f"{coeff['estimate']:.1f}"
    for method, name in (("own", "Own"), ("panel", "Panel")):
        gap = speed["adjusted_slopes"][method]["men_minus_women_pp"]["attr"]
        values["PboLooksGap" + name] = f"{gap['estimate']:.1f}"
        values["PboLooksInterval" + name] = interval(gap)
    gap = speed["quintile_contrast_men_minus_women"]
    values["PboQuintileGap"] = f"{100 * gap['estimate']:.1f}"
    values["PboQuintileInterval"] = interval(gap, 100)

    # New Voice, Dyadic Spark, Pair-Bond Love Reversal, Joel 2020, SCCS & Life-Course macros
    swv = pov["self_written_voice"]
    dc = speed["dyadic_chemistry"]
    pbr = kow["pair_bond_reversal"]
    prqc_b = pbr["relationship_quality_prqc"]
    inf_b = pbr["reported_infidelity"]
    tla_b = kow["triangular_love_arc"]
    mwl_b = kow["marry_without_love"]
    sma_b = kow["social_media_and_alternatives"]
    joel_b = osyn["joel_2020"]
    sccs_b = osyn["betzig_sccs"]
    jf_b = osyn["published_benchmarks"]["jankowiak_fischer_1992_love"]
    wz_map = {z["zone_key"]: z for z in fa["zone_drivers_and_rejectors"]["Female"]["zones"]}
    mz_map = {z["zone_key"]: z for z in fa["zone_drivers_and_rejectors"]["Male"]["zones"]}

    values.update({
        "PboPovCrossAge": f"{pov['childless_crossover']['band_midpoint_interpolated_age']:.1f}",
        "PboPovCrossExact": str(pov["childless_crossover"]["first_nonpositive_exact_age"]),
        "PboVoiceEssaysN": f"{swv['eligible_with_essay']['Pooled']:,}",
        "PboVoiceEssaysW": f"{swv['eligible_with_essay']['Female']:,}",
        "PboVoiceEssaysM": f"{swv['eligible_with_essay']['Male']:,}",
        "PboVoiceFamilyW": format_rate(swv["themes"]["family_kids_home"]["Female"]["share"]),
        "PboVoiceFamilyM": format_rate(swv["themes"]["family_kids_home"]["Male"]["share"]),
        "PboVoiceLaughW": format_rate(swv["themes"]["laugh_humor_smile"]["Female"]["share"]),
        "PboVoiceLaughM": format_rate(swv["themes"]["laugh_humor_smile"]["Male"]["share"]),
        "PboVoiceKindW": format_rate(swv["themes"]["kind_caring_loving"]["Female"]["share"]),
        "PboVoiceKindM": format_rate(swv["themes"]["kind_caring_loving"]["Male"]["share"]),
        "PboVoiceHonestW": format_rate(swv["themes"]["honest_loyal_trust"]["Female"]["share"]),
        "PboVoiceHonestM": format_rate(swv["themes"]["honest_loyal_trust"]["Male"]["share"]),
        "PboVoiceTallW": f"{100.0 * swv['themes']['tall_6ft_mention']['Female']['share']:.2f}\\%",
        "PboVoiceRichW": f"{100.0 * swv['themes']['rich_six_figure']['Female']['share']:.2f}\\%",
        "PboVoiceFamilyTallRatio": f"{swv['voice_to_checkbox_ratios']['female_family_to_tall']:.0f}",
        "PboVoiceHonestTallRatio": f"{swv['voice_to_checkbox_ratios']['female_honesty_to_tall']:.0f}",
        "PboVoiceKindRichRatio": f"{swv['voice_to_checkbox_ratios']['female_kindness_to_wealth']:.0f}",
        "PboDyadLikeN": f"{dc['variance_partition']['like']['n']:,}",
        "PboDyadPairsN": f"{dc['mutual_dyadic_reciprocity_like']['n_dyads']:,}",
        "PboDyadConsensusShare": format_rate(dc["variance_partition"]["like"]["unbiased_srm_shares"]["partner_consensus"]),
        "PboDyadChooserShare": format_rate(dc["variance_partition"]["like"]["unbiased_srm_shares"]["chooser_leniency"]),
        "PboDyadUniqueShare": format_rate(dc["variance_partition"]["like"]["unbiased_srm_shares"]["unique_dyadic_and_noise"]),
        "PboDyadLooksConsensus": format_rate(dc["variance_partition"]["attr"]["unbiased_srm_shares"]["partner_consensus"]),
        "PboDyadSincConsensus": format_rate(dc["variance_partition"]["sinc"]["unbiased_srm_shares"]["partner_consensus"]),
        "PboDyadIntelConsensus": format_rate(dc["variance_partition"]["intel"]["unbiased_srm_shares"]["partner_consensus"]),
        "PboDyadSharConsensus": format_rate(dc["variance_partition"]["shar"]["unbiased_srm_shares"]["partner_consensus"]),
        "PboDyadReciprocityR": f"{dc['mutual_dyadic_reciprocity_like']['estimate']:+.2f}",
        "PboDyadSparkSlopeW": f"{dc['reciprocity_adjusted_slopes']['women']['dyadic_spark_sum_fun_shar_prob_pp']['estimate']:+.1f}",
        "PboDyadLooksSlopeW": f"{dc['reciprocity_adjusted_slopes']['women']['coefficients_pp']['attr']['estimate']:+.1f}",
        "PboDyadChosenBackW": f"{dc['reciprocity_adjusted_slopes']['women']['coefficients_pp']['prob']['estimate']:+.1f}",
        "PboDyadChosenBackM": f"{dc['reciprocity_adjusted_slopes']['men']['coefficients_pp']['prob']['estimate']:+.1f}",
        "PboDoorN": f"{kow['partnering_slopes']['n']:,}",
        "PboDoorMoneyM": f"{kow['partnering_slopes']['by_sex']['men']['resources']['pp_per_sd']:+.2f}",
        "PboDoorLooksM": f"{kow['partnering_slopes']['by_sex']['men']['looks']['pp_per_sd']:+.2f}",
        "PboDoorLooksW": f"{kow['partnering_slopes']['by_sex']['women']['looks']['pp_per_sd']:+.2f}",
        "PboDoorKindM": f"{kow['partnering_slopes']['by_sex']['men']['kindness']['pp_per_sd']:+.2f}",
        "PboDoorKindW": f"{kow['partnering_slopes']['by_sex']['women']['kindness']['pp_per_sd']:+.2f}",
        "PboBondPrqcN": f"{prqc_b['n']:,}",
        "PboBondInfN": f"{inf_b['n']:,}",
        "PboBondKindW": f"{prqc_b['by_sex']['women']['traits']['kindness']['sd_per_sd']:+.2f}",
        "PboBondKindM": f"{prqc_b['by_sex']['men']['traits']['kindness']['sd_per_sd']:+.2f}",
        "PboBondMoneyW": f"{prqc_b['by_sex']['women']['traits']['resources']['sd_per_sd']:+.2f}",
        "PboBondMoneyM": f"{prqc_b['by_sex']['men']['traits']['resources']['sd_per_sd']:+.2f}",
        "PboBondInfKindW": f"{inf_b['by_sex']['women']['traits']['kindness']['pp_per_sd']:+.1f}",
        "PboBondInfKindM": f"{inf_b['by_sex']['men']['traits']['kindness']['pp_per_sd']:+.1f}",
        "PboBondInfMoneyW": f"{inf_b['by_sex']['women']['traits']['resources']['pp_per_sd']:+.1f}",
        "PboJoelDatasetsN": str(joel_b["satisfaction"]["distinct_datasets"]),
        "PboJoelRelBaseRTwo": f"{joel_b['satisfaction']['models']['ActorRelBaseline']['unweighted_mean_r2_pct']:.1f}\\%",
        "PboJoelTraitBaseRTwo": f"{joel_b['satisfaction']['models']['PartnerTraitBaseline']['unweighted_mean_r2_pct']:.1f}\\%",
        "PboJoelBaseRatio": f"{joel_b['satisfaction']['actor_rel_to_partner_trait_baseline_ratio']:.1f}",
        "PboJoelRelFollowRTwo": f"{joel_b['satisfaction']['models']['ActorRelFollowup']['unweighted_mean_r2_pct']:.1f}\\%",
        "PboJoelTraitFollowRTwo": f"{joel_b['satisfaction']['models']['PartnerTraitFollowup']['unweighted_mean_r2_pct']:.1f}\\%",
        "PboLoveArcN": f"{tla_b['n_total']:,}",
        "PboLoveCommStart": f"{tla_b['by_duration']['<=1y']['commitment']['mean']:.2f}",
        "PboLoveCommEnd": f"{tla_b['by_duration']['25y+']['commitment']['mean']:.2f}",
        "PboLoveIntimEnd": f"{tla_b['by_duration']['25y+']['intimacy']['mean']:.2f}",
        "PboLovePassStart": f"{tla_b['by_duration']['<=1y']['passion']['mean']:.2f}",
        "PboLovePassEnd": f"{tla_b['by_duration']['25y+']['passion']['mean']:.2f}",
        "PboMarryNoLoveN": f"{mwl_b['n_total']:,}",
        "PboMarryNoLoveMedW": f"{mwl_b['by_sex']['women']['median']:.0f}",
        "PboMarryNoLoveMedM": f"{mwl_b['by_sex']['men']['median']:.0f}",
        "PboAssortReligR": f"{kow['assortative_mating']['religion']['pearson_r']:.2f}",
        "PboAssortKindR": f"{kow['assortative_mating']['kindness']['pearson_r']:.2f}",
        "PboSocMedInfSlope": f"{sma_b['slopes_per_daily_hour']['infidelity_pp_per_hour']['estimate']:+.2f}",
        "PboSocMedPrqcSlope": f"{sma_b['slopes_per_daily_hour']['prqc_sd_per_hour']['estimate']:+.3f}",
        "PboSocMedQoaSlope": f"{sma_b['slopes_per_daily_hour']['qoa_sd_per_hour']['estimate']:+.3f}",
        "PboSocMedHighInf": f"{sma_b['by_qoa_tier']['high_4.5_to_7.0']['infidelity_rate_pct']['mean']:.1f}\\%",
        "PboSocMedLowInf": f"{sma_b['by_qoa_tier']['low_1_to_2.5']['infidelity_rate_pct']['mean']:.1f}\\%",
        "PboBetzigSocN": str(sccs_b["causes"]["adultery_infidelity"]["coded_societies"]),
        "PboBetzigAdulteryPct": format_rate(sccs_b["causes"]["adultery_infidelity"]["share_mentioned"]),
        "PboBetzigCrueltyPct": format_rate(sccs_b["causes"]["cruelty_maltreatment"]["share_mentioned"]),
        "PboBetzigAgingPct": format_rate(sccs_b["causes"]["old_age_loss_of_looks"]["share_mentioned"]),
        "PboBetzigRatio": f"{sccs_b['adultery_to_aging_looks_ratio']:.1f}",
        "PboJankowiakPct": format_rate(jf_b["share_documented"]),
        "PboWhyteEmotConnW": f"{whyte['stated_importance']['pooled']['Women']['emotional_connection']['mean']:.1f}",
        "PboWhyteEmotConnM": f"{whyte['stated_importance']['pooled']['Men']['emotional_connection']['mean']:.1f}",
        "PboWhyteHappyLiftW": f"{whyte['happiness_by_partnered']['Women']['raw_lift_pts']['estimate']:+.1f}",
        "PboWhyteHappyLiftM": f"{whyte['happiness_by_partnered']['Men']['raw_lift_pts']['estimate']:+.1f}",
        "PboAgeScissorsOld": f"{kow['partner_age_gap_by_age']['brackets']['60-79']['women_minus_men_years']:.1f}",
        "PboFaceEyesDrvW": f"{wz_map['eyes_iris']['top15_driver_pct']:+.2f}",
        "PboFaceMouthDrvW": f"{wz_map['lips_vermilion']['top15_driver_pct']:+.2f}",
        "PboFaceEyesDrvM": f"{mz_map['eyes_iris']['top15_driver_pct']:+.2f}",
        "PboFaceMouthDrvM": f"{mz_map['lips_vermilion']['top15_driver_pct']:+.2f}",
    })

    # v13 Cross-Scale Causal Closure, NHS Campfire, Hobby/Hindsight, HCMST, Checklist/Marrying-Up & WORDSUM macros
    cc_b = osyn["childhood_coevolution"]
    sccs_cc_b = cc_b["sccs_alloparenting_and_sharing"]
    e22_b = cc_b["proposition_22_e22_coevolution"]
    nhs_b = osyn["nhs_campfire"]
    alg_b = nhs_b["acoustic_legibility_gradient"]
    chk_sd_b = speed["checklist_and_hindsight"]
    hw_b = chk_sd_b["hobby_checklist_vs_live_wavelength"]
    wps_b = chk_sd_b["within_person_shifts"]
    fup_b = chk_sd_b["post_event_followup_3_4_weeks"]
    hcm_b = osyn["hcmst_2017"]
    chk_kow_b = kow["checklist_and_surplus_decomposition"]
    pk_kow_b = chk_kow_b["pooled"]["predictors_sd_per_sd"]
    id_kow_b = kow["short_vs_long_term_ideals"]["by_sex"]["women"]
    jl_kow_b = kow["jealousy_asymmetry"]["by_sex"]["women"]
    gss_all_b = osyn["v9_anchors"]["gss_three_episodes_all"]

    values.update({
        "PboChildPeakRmr": f"{cc_b['kuzawa_2014_brain_metabolism']['peak_rmr_pct']:.1f}\\%",
        "PboChildPeakDer": f"{cc_b['kuzawa_2014_brain_metabolism']['peak_der_pct']:.1f}\\%",
        "PboChildAdultRmrRatio": f"{cc_b['kuzawa_2014_brain_metabolism']['child_to_adult_rmr_ratio']:.1f}",
        "PboSccsChildSoloMomPct": format_rate(sccs_cc_b["early_childhood_exclusive_mother_sccs52"]["share"]),
        "PboSccsChildAlloMajPct": format_rate(sccs_cc_b["early_childhood_nonmaternal_majority_sccs52"]["share"]),
        "PboSccsFatherInfPct": format_rate(sccs_cc_b["father_close_infancy_sccs53"]["share"]),
        "PboSccsFatherChildPct": format_rate(sccs_cc_b["father_close_early_childhood_sccs54"]["share"]),
        "PboSccsFatherOccPct": format_rate(sccs_cc_b["father_occasional_or_closer_early_childhood_sccs54"]["share"]),
        "PboSccsFoodSharePct": format_rate(sccs_cc_b["community_food_sharing_beyond_nuclear_sccs1718"]["share"]),
        "PboETwentyTwoSolo": f"{e22_b['regimes']['solitary_ape_mother']['equilibrium_cm3_closed_form']:,.0f}",
        "PboETwentyTwoRawAllo": f"{e22_b['regimes']['early_homo_allomaternal_raw']['equilibrium_cm3_closed_form']:,.0f}",
        "PboETwentyTwoCookNoLove": f"{e22_b['regimes']['cooking_without_pair_bond_love']['equilibrium_cm3_closed_form']:,.0f}",
        "PboETwentyTwoFull": f"{e22_b['regimes']['late_homo_cooking_plus_pair_bond_love']['equilibrium_cm3_closed_form']:,.0f}",
        "PboETwentyTwoLoveGain": f"{e22_b['love_devotion_gain_cm3']:+.0f}",
        "PboNhsSongsN": str(nhs_b["discography_songs"]),
        "PboNhsSocN": str(nhs_b["discography_societies"]),
        "PboNhsRatersN": f"{nhs_b['naive_listener_raters_published']:,}",
        "PboNhsEthPerfN": f"{nhs_b['ethnography_performances']:,}",
        "PboNhsLullabyD": f"{alg_b['lullaby']['cohen_d']:+.2f}",
        "PboNhsDanceD": f"{alg_b['dance']['cohen_d']:+.2f}",
        "PboNhsHealD": f"{alg_b['healing']['cohen_d']:+.2f}",
        "PboNhsLoveD": f"{alg_b['love']['cohen_d']:+.2f}",
        "PboNhsLullabyLoveRatio": f"{nhs_b['lullaby_to_love_d_ratio']:.1f}",
        "PboNhsDanceLoveRatio": f"{nhs_b['dance_to_love_d_ratio']:.1f}",
        "PboNhsDanceShare": format_rate(nhs_b["ethnographic_hearth_contexts"]["any_communal_dance"]["share_of_coded"]),
        "PboHobbyDyadsN": f"{hw_b['n_common_dyad_decisions']:,}",
        "PboHobbyObjR": f"{hw_b['r_objective_hobby_corr_yes']['estimate']:+.3f}",
        "PboHobbyLiveR": f"{hw_b['r_live_shared_wavelength_yes']['estimate']:+.3f}",
        "PboHobbyRatio": f"{hw_b['live_to_checklist_r_ratio']:.1f}",
        "PboHobbyFeSlopeW": f"{hw_b['chooser_fe_slopes']['women']['coefficients_pp']['int_corr']['estimate']:+.1f}",
        "PboHobbyLiveSlopeW": f"{hw_b['chooser_fe_slopes']['women']['coefficients_pp']['shar']['estimate']:+.1f}",
        "PboHindLooksJumpW": f"{wps_b['nextday_realized_minus_pre_stated']['women']['delta_points']['attr']['estimate']:+.1f}",
        "PboHindIntelDropW": f"{wps_b['nextday_realized_minus_pre_stated']['women']['delta_points']['intel']['estimate']:+.1f}",
        "PboHindIntelSnapW": f"{wps_b['threeweek_stated_minus_nextday_realized']['women']['delta_points']['intel']['estimate']:+.1f}",
        "PboHindLooksRecedeW": f"{wps_b['threeweek_stated_minus_nextday_realized']['women']['delta_points']['attr']['estimate']:+.1f}",
        "PboMatchDatePct": format_rate(fup_b["went_on_followup_date_share_among_matched"]),
        "PboMatchCallPct": format_rate(fup_b["called_or_was_called_share_among_matched"]),
        "PboHcmstN": f"{hcm_b['total_respondents']:,}",
        "PboHcmstCouplesN": f"{hcm_b['meeting_coded_couples']:,}",
        "PboHcmstAppMeanYr": f"{hcm_b['by_venue']['online_dating_app']['pre_romance_acquaintance_years']['mean']:.2f}",
        "PboHcmstAppMedYr": f"{hcm_b['by_venue']['online_dating_app']['pre_romance_acquaintance_years']['median']:.2f}",
        "PboHcmstAppZeroPct": format_rate(hcm_b["by_venue"]["online_dating_app"]["pre_romance_acquaintance_years"]["share_zero_years"]),
        "PboHcmstAppZeroTiePct": format_rate(hcm_b["by_venue"]["online_dating_app"]["zero_prior_social_tie_share"]),
        "PboHcmstInstMeanYr": f"{hcm_b['institutional_and_kin_pooled']['pre_romance_acquaintance_years']['mean']:.2f}",
        "PboHcmstSchoolMeanYr": f"{hcm_b['by_venue']['school_college']['pre_romance_acquaintance_years']['mean']:.2f}",
        "PboHcmstAuditRatio": f"{hcm_b['audit_compression_ratios']['institutional_kin_to_app_mean_ratio']:.1f}",
        "PboHcmstSchoolRatio": f"{hcm_b['audit_compression_ratios']['school_to_app_mean_ratio']:.1f}",
        "PboHcmstPreEightyYr": f"{hcm_b['historical_eras']['pre_1980']['mean_pre_romance_years']:.2f}",
        "PboHcmstRecentYr": f"{hcm_b['historical_eras']['2012_2017']['mean_pre_romance_years']:.2f}",
        "PboHcmstRecentOnlinePct": format_rate(hcm_b["historical_eras"]["2012_2017"]["online_all_share"]),
        "PboHcmstRecentAppPct": format_rate(hcm_b["historical_eras"]["2012_2017"]["online_dating_app_share"]),
        "PboHcmstQualApp": f"{hcm_b['by_venue']['online_dating_app']['intact_relationship_quality_1_to_5']['mean']:.2f}",
        "PboHcmstQualInst": f"{hcm_b['institutional_and_kin_pooled']['intact_relationship_quality_1_to_5']['mean']:.2f}",
        "PboHcmstQualAdjDiff": f"{hcm_b['intact_quality_boundary_check']['online_app_minus_offline_pts_1_to_5']:+.2f}",
        "PboChkCouplesN": f"{chk_kow_b['n_total']:,}",
        "PboChkCountriesN": str(chk_kow_b["pooled"]["countries"]),
        "PboChkKindSD": f"{pk_kow_b['partner_kindness']['estimate']:+.2f}",
        "PboChkLooksSD": f"{pk_kow_b['partner_looks']['estimate']:+.2f}",
        "PboChkFitSD": f"{pk_kow_b['idiographic_checklist_fit']['estimate']:+.2f}",
        "PboChkResSD": f"{pk_kow_b['partner_resources']['estimate']:+.2f}",
        "PboChkMarryUpSD": f"{pk_kow_b['marrying_up_surplus']['estimate']:+.2f}",
        "PboChkMarryUpFiveSD": f"{chk_kow_b['five_trait_controlled_marrying_up_sd']:+.2f}",
        "PboChkKindFitRatio": f"{chk_kow_b['pooled']['kindness_to_checklist_ratio']:.1f}",
        "PboIdealStKindW": f"{id_kow_b['kindness']['short_term_mean_1_to_11']:.2f}",
        "PboIdealLtKindW": f"{id_kow_b['kindness']['long_term_mean_1_to_11']:.2f}",
        "PboIdealStLooksW": f"{id_kow_b['looks']['short_term_mean_1_to_11']:.2f}",
        "PboJealEmoW": f"{jl_kow_b['emotional_1_to_7']['mean']:.2f}",
        "PboJealSexW": f"{jl_kow_b['sexual_1_to_7']['mean']:.2f}",
        "PboJealDiffW": f"{jl_kow_b['emotional_minus_sexual']['mean']:+.2f}",
        "PboGssVocPairM": f"{gss_all_b['vocabulary']['men']['pairing'][0]:+.3f}",
        "PboGssVocNumM": f"{gss_all_b['vocabulary']['men']['number'][0]:+.3f}",
        "PboGssSchPairM": f"{gss_all_b['schooling']['men']['pairing'][0]:+.3f}",
        "PboGssPrePairM": f"{gss_all_b['prestige']['men']['pairing'][0]:+.3f}",
    })

    # v14 Empirical Mind Selection (Speed-Dating Objective SAT) & OkCupid-JSE Audience Staging macros
    sat_b = chk_sd_b["objective_sat_blindness"]
    okc_b = osyn["okcupid_jse"]
    ost_b = okc_b["orientation_staging_experiment"]
    tpp_b = okc_b["ten_prompt_portfolio"]
    mdv_b = tpp_b["multi_domain_lexical_validity"]
    amr_b = tpp_b["assortative_mind_requests_essay9"]
    e9_b = tpp_b["essay9_message_me_if"]
    e3_b = tpp_b["essay3_first_noticed"]
    sfc_b = okc_b["sf_childless_fertility_crossover"]

    values.update({
        "PboThreePlatformRecords": f"{pov['accounts']['rows'] + hon['inventory']['canonical_profiles'] + okc_b['total_profiles']:,}",
        "PboSatPartN": str(sat_b["n_participants_with_sat"]),
        "PboSatDatesN": f"{sat_b['n_dates_with_partner_sat']:,}",
        "PboSatSingleR": f"{sat_b['r_stranger_intel_vs_partner_sat']['estimate']:+.3f}",
        "PboSatSingleRTwo": f"{100.0 * sat_b['stranger_intel_sat_r2']:.2f}\\%",
        "PboSatPanelR": f"{sat_b['r_panel_intel_vs_partner_sat']['estimate']:+.3f}",
        "PboSatPanelRTwo": f"{100.0 * sat_b['panel_intel_sat_r2']:.2f}\\%",
        "PboSatPanelGain": f"{sat_b['panel_to_single_r2_ratio']:.1f}",
        "PboSatYesR": f"{sat_b['r_partner_sat_vs_yes']['estimate']:+.3f}",
        "PboSatMatchR": f"{sat_b['r_matched_couple_sat']:+.3f}",
        "PboOkcN": f"{okc_b['total_profiles']:,}",
        "PboOkcMenN": f"{okc_b['sex_counts']['men']:,}",
        "PboOkcWomenN": f"{okc_b['sex_counts']['women']:,}",
        "PboOkcEssaysN": f"{tpp_b['total_nonempty_essays_ge5_chars']:,}",
        "PboOkcStrMenN": f"{ost_b['groups']['men_straight']['n']:,}",
        "PboOkcGayMenN": f"{ost_b['groups']['men_gay']['n']:,}",
        "PboOkcStrWomenN": f"{ost_b['groups']['women_straight']['n']:,}",
        "PboOkcStrJump": f"{ost_b['groups']['men_straight']['stature']['jump_5ft11_to_6ft0_pct']:+.1f}\\%",
        "PboOkcStrCliff": f"{abs(ost_b['groups']['men_straight']['stature']['drop_6ft0_to_6ft1_pct']):.1f}\\%",
        "PboOkcGayJump": f"{ost_b['groups']['men_gay']['stature']['jump_5ft11_to_6ft0_pct']:+.1f}\\%",
        "PboOkcIncStrM": format_rate(ost_b["groups"]["men_straight"]["income_disclosure"]["disclosed_positive_share"]),
        "PboOkcIncGayM": format_rate(ost_b["groups"]["men_gay"]["income_disclosure"]["disclosed_positive_share"]),
        "PboOkcIncStrW": format_rate(ost_b["groups"]["women_straight"]["income_disclosure"]["disclosed_positive_share"]),
        "PboOkcIncGayRatio": f"{ost_b['straight_to_gay_male_income_disclosure_ratio']:.2f}",
        "PboOkcIncFemRatio": f"{ost_b['straight_male_to_female_income_disclosure_ratio']:.2f}",
        "PboOkcDomWritersN": f"{mdv_b['n_complete_9_domain_writers']:,}",
        "PboOkcDomROne": f"{mdv_b['single_prompt_reliability_r1']:.3f}",
        "PboOkcDomRNine": f"{mdv_b['spearman_brown_curve_k_1_to_9']['9']:.3f}",
        "PboOkcDomGain": f"{mdv_b['k9_to_k1_validity_ratio']:.1f}",
        "PboOkcAssortN": f"{amr_b['n_users_ge3_domains_with_essay9']:,}",
        "PboOkcAssortLo": format_rate(amr_b["by_lexical_quintile"]["Q1"]["ask_mind_curiosity_share"]),
        "PboOkcAssortHi": format_rate(amr_b["by_lexical_quintile"]["Q5"]["ask_mind_curiosity_share"]),
        "PboOkcAssortRatio": f"{amr_b['q5_to_q1_mind_request_ratio']:.2f}",
        "PboOkcPrivPct": format_rate(tpp_b["prompts"]["essay8"]["share"]),
        "PboOkcSelfPct": format_rate(tpp_b["prompts"]["essay0"]["share"]),
        "PboOkcMsgN": f"{e9_b['n_answered']:,}",
        "PboOkcMsgHumor": format_rate(e9_b["themes"]["humor_banter"]["share"]),
        "PboOkcMsgKind": format_rate(e9_b["themes"]["kindness_sincerity"]["share"]),
        "PboOkcMsgMind": format_rate(e9_b["themes"]["curiosity_intelligence"]["share"]),
        "PboOkcMsgTall": f"{100.0 * e9_b['themes']['tall_6ft_height']['share']:.2f}\\%",
        "PboOkcMsgRich": f"{100.0 * e9_b['themes']['rich_wealthy_sixfig']['share']:.2f}\\%",
        "PboOkcHumorTallRatio": f"{e9_b['ratios']['humor_to_tall']:.1f}",
        "PboOkcHumorRichRatio": f"{e9_b['ratios']['humor_to_wealth']:.1f}",
        "PboOkcNoticeN": f"{e3_b['n_answered']:,}",
        "PboOkcSmilePct": format_rate(e3_b["themes"]["smile_laugh"]["share"]),
        "PboOkcEyesPct": format_rate(e3_b["themes"]["eyes"]["share"]),
        "PboOkcHeightNoticePct": format_rate(e3_b["themes"]["height"]["share"]),
        "PboOkcSfCrossAge": f"{sfc_b['interpolated_crossover_age']:.1f}",
        "PboOkcSfGapYoung": format_pp(sfc_b["by_age_band"]["25-29"]["women_minus_men_pp"]),
        "PboOkcSfGapOlder": format_pp(sfc_b["by_age_band"]["45-54"]["women_minus_men_pp"]),
    })

    # v15 Primary vs. Secondary Mating Criteria, Cross-Source Male-Female Dynamics & Proposition 23 (E23) macros
    csm_b = chk_sd_b["cross_sex_misperception"]
    sspi_b = chk_sd_b["same_sex_pluralistic_ignorance"]
    init_b = fup_b["initiation_by_sex"]
    hvk_b = kow["homophily_vs_kindness"]
    rsk_b = kow["reverence_vs_settling_kink"]["by_sex"]
    ces_b = kow["courtship_effort_and_soi_lifecycle"]
    stg_b = ces_b["by_relationship_stage"]["by_sex"]
    crh_b = hcm_b["cohabitation_rush_hazard"]
    bia_b = hcm_b["breakup_initiation_asymmetry"]
    avc_b = swv["action_verb_calendar"]
    sar_b = pov["second_act_reentry"]
    e23_b = osyn["proposition_23_e23"]

    w_inf_rev_drop = (
        rsk_b["women"]["tiers_by_trait"]["looks"]["settled_mate_lt_self"]["infidelity_rate_pct"]["mean"]
        - rsk_b["women"]["tiers_by_trait"]["looks"]["revered_mate_gt_self"]["infidelity_rate_pct"]["mean"]
    )
    m_inf_rev_drop = (
        rsk_b["men"]["tiers_by_trait"]["looks"]["settled_mate_lt_self"]["infidelity_rate_pct"]["mean"]
        - rsk_b["men"]["tiers_by_trait"]["looks"]["revered_mate_gt_self"]["infidelity_rate_pct"]["mean"]
    )

    values.update({
        "PboMindReadLooksW": f"{csm_b['women_about_men']['delta_points']['attr']['estimate']:+.1f}",
        "PboMindReadIntelW": f"{csm_b['women_about_men']['delta_points']['intel']['estimate']:+.1f}",
        "PboMindReadSincW": f"{csm_b['women_about_men']['delta_points']['sinc']['estimate']:+.1f}",
        "PboMindReadLooksM": f"{csm_b['men_about_women']['delta_points']['attr']['estimate']:+.1f}",
        "PboMindReadIntelM": f"{csm_b['men_about_women']['delta_points']['intel']['estimate']:+.1f}",
        "PboMindReadSincM": f"{csm_b['men_about_women']['delta_points']['sinc']['estimate']:+.1f}",
        "PboPeerIgnorLooksW": f"{sspi_b['women']['delta_points']['attr']['estimate']:+.1f}",
        "PboPeerIgnorLooksM": f"{sspi_b['men']['delta_points']['attr']['estimate']:+.1f}",
        "PboCallRatioMF": f"{init_b['male_to_female_calls_per_match_ratio']:.1f}",
        "PboZeroCallsW": format_rate(init_b["women"]["zero_calls_placed_share"]),
        "PboZeroCallsM": format_rate(init_b["men"]["zero_calls_placed_share"]),
        "PboHomoPolR": f"{hvk_b['politics']['assortative_pearson_r']:+.2f}",
        "PboHomoClsR": f"{hvk_b['social_class']['assortative_pearson_r']:+.2f}",
        "PboHomoEduR": f"{hvk_b['education']['assortative_pearson_r']:+.2f}",
        "PboHomoPolSlope": f"{hvk_b['politics']['homophily_prqc_sd_per_sd']['estimate']:+.2f}",
        "PboHomoClsSlope": f"{hvk_b['social_class']['homophily_prqc_sd_per_sd']['estimate']:+.2f}",
        "PboHomoEduSlope": f"{hvk_b['education']['homophily_prqc_sd_per_sd']['estimate']:+.2f}",
        "PboHomoPolRatio": f"{hvk_b['politics']['kindness_to_homophily_ratio']:.1f}",
        "PboHomoClsRatio": f"{hvk_b['social_class']['kindness_to_homophily_ratio']:.1f}",
        "PboHomoEduRatio": f"{hvk_b['education']['kindness_to_homophily_ratio']:.1f}",
        "PboSettleKindSlopeW": f"{rsk_b['women']['country_fe_spline_sd_per_pt']['settling_deficit_kindness']['estimate']:+.2f}",
        "PboSettleKindSlopeM": f"{rsk_b['men']['country_fe_spline_sd_per_pt']['settling_deficit_kindness']['estimate']:+.2f}",
        "PboRevKindSlopeW": f"{rsk_b['women']['country_fe_spline_sd_per_pt']['reverence_surplus_kindness']['estimate']:+.2f}",
        "PboRevKindSlopeM": f"{rsk_b['men']['country_fe_spline_sd_per_pt']['reverence_surplus_kindness']['estimate']:+.2f}",
        "PboSettleRevRatioW": f"{rsk_b['women']['kindness_settling_to_reverence_slope_ratio']:.1f}",
        "PboSettleRevRatioM": f"{rsk_b['men']['kindness_settling_to_reverence_slope_ratio']:.1f}",
        "PboRevResSlopeW": f"{rsk_b['women']['country_fe_spline_sd_per_pt']['reverence_surplus_resources']['estimate']:+.2f}",
        "PboRevResSlopeM": f"{rsk_b['men']['country_fe_spline_sd_per_pt']['reverence_surplus_resources']['estimate']:+.2f}",
        "PboRevLooksInfDropW": f"{w_inf_rev_drop:.1f}",
        "PboRevLooksInfDropM": f"{m_inf_rev_drop:.1f}",
        "PboEffortDatingW": f"{stg_b['women']['dating']['total_effort_min']['mean']:.0f}",
        "PboEffortMarriedW": f"{stg_b['women']['married']['total_effort_min']['mean']:.0f}",
        "PboEffortDropPctW": f"{abs(stg_b['women']['dating_to_married_drop_pct']):.1f}\\%",
        "PboEffortDatingM": f"{stg_b['men']['dating']['total_effort_min']['mean']:.0f}",
        "PboEffortMarriedM": f"{stg_b['men']['married']['total_effort_min']['mean']:.0f}",
        "PboEffortDropPctM": f"{abs(stg_b['men']['dating_to_married_drop_pct']):.1f}\\%",
        "PboSoiRatioYoung": f"{ces_b['by_age_bracket']['18-24']['men_to_women_soi_desire_ratio']:.2f}",
        "PboSoiRatioOld": f"{ces_b['by_age_bracket']['60-79']['men_to_women_soi_desire_ratio']:.2f}",
        "PboCohabRushPct": format_rate(crh_b["by_romance_to_cohab_window"]["le_3m_rush"]["breakup_share"]),
        "PboCohabSlowPct": format_rate(crh_b["by_romance_to_cohab_window"]["gt_2_5y"]["breakup_share"]),
        "PboCohabRushRatio": f"{crh_b['le_3m_to_gt_2_5y_breakup_ratio']:.1f}",
        "PboCohabRushAdjPP": f"{crh_b['cohort_age_adjusted_rush_le_3m_pp']['estimate']:+.1f}",
        "PboDivUnilatW": format_rate(bia_b["married_divorce_or_separation"]["woman_share_of_unilateral"]),
        "PboDivHusRepW": format_rate(bia_b["married_divorce_or_separation"]["by_respondent_gender_unilateral"]["male_respondents_wife_share"]),
        "PboNonMarUnilatW": format_rate(bia_b["nonmarital_breakup"]["woman_share_of_unilateral"]),
        "PboVerbCookYoungW": format_rate(avc_b["by_age_bracket"]["18-22"]["cook_bake_hearth"]["Female"]["share"]),
        "PboVerbCookOldW": format_rate(avc_b["by_age_bracket"]["65-79"]["cook_bake_hearth"]["Female"]["share"]),
        "PboVerbCookRiseW": f"{avc_b['lifespan_shifts_18_22_to_65_79']['female_cook_rise_ratio']:.1f}",
        "PboVerbCookYoungM": format_rate(avc_b["by_age_bracket"]["18-22"]["cook_bake_hearth"]["Male"]["share"]),
        "PboVerbCookOldM": format_rate(avc_b["by_age_bracket"]["65-79"]["cook_bake_hearth"]["Male"]["share"]),
        "PboVerbCookRiseM": f"{avc_b['lifespan_shifts_18_22_to_65_79']['male_cook_rise_ratio']:.1f}",
        "PboVerbPartyDrop": f"{abs(avc_b['lifespan_shifts_18_22_to_65_79']['pooled_party_drop_pct']):.1f}\\%",
        "PboDivMarryDropW": f"{abs(sar_b['headline_contrasts']['divorced_vs_single_female_marry_change_pct_40_49']):.1f}\\%",
        "PboDivSeriousMidW": format_rate(sar_b["by_age_window"]["40-49"]["divorced"]["Female"]["serious_commitment"]["share"]),
        "PboSinMarryMidW": format_rate(sar_b["by_age_window"]["40-49"]["single"]["Female"]["marry_intent"]["share"]),
        "PboDivMarryMidW": format_rate(sar_b["by_age_window"]["40-49"]["divorced"]["Female"]["marry_intent"]["share"]),
        "PboWidMarryMidM": format_rate(sar_b["by_age_window"]["40-49"]["widowed"]["Male"]["marry_intent"]["share"]),
        "PboWidMarryMidW": format_rate(sar_b["by_age_window"]["40-49"]["widowed"]["Female"]["marry_intent"]["share"]),
        "PboWidMarrySenM": format_rate(sar_b["by_age_window"]["60-75"]["widowed"]["Male"]["marry_intent"]["share"]),
        "PboWidMarrySenW": format_rate(sar_b["by_age_window"]["60-75"]["widowed"]["Female"]["marry_intent"]["share"]),
        "PboETwentyThreeDropPct": f"{abs(e23_b['part_i_staging_moral_hazard']['drop_pct_closed_form']):.1f}\\%",
        "PboETwentyThreeDivW": format_rate(e23_b["part_iii_unilateral_divorce_flip"]["married_woman_unilateral_closed_form"]),
    })

    for name in values:
        assert name.isalpha(), f"LaTeX macro name must be alphabetic: {name}"
    return "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())


def pov_tables(pov):
    ages = ("25-29", "45-49")
    primary = [r"Age & Recorded gender & Profiles & Want & Do not want & Undecided/Open & Nonresponse\\\midrule"]
    for age in ages:
        for sex, name, _, _ in SERIES:
            row = pov["sensitivity"][age]["main"]["responses"][sex]
            cells = [format_rate(row["shares"][k]) for k in ("want", "do_not_want", "undecided_open", "nonresponse")]
            primary.append(" & ".join([age.replace("-", "--"), name, f"{row['n']:,}", *cells]) + r"\\")
    robust = [r"& \multicolumn{3}{c}{Ages 25--29} & \multicolumn{3}{c}{Ages 45--49}\\",
              r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
              r"Cohort or weighting & Women & Men & W $-$ M & Women & Men & W $-$ M\\\midrule"]
    variants = [("main", "Main cohort"), ("exact_age", "Standardized by exact age"),
                ("single", "Marked single"), ("all_seeking", "All seeking records"),
                ("answered", "Answered only"), ("single_answered_exact_age", "Single, answered, age-standardized")]
    for key, label in variants:
        row = [label]
        for age in ages:
            cell = pov["sensitivity"][age][key]
            shares = cell["want_shares"]
            row.extend([format_rate(shares["Female"]), format_rate(shares["Male"]),
                        format_pp(cell["women_minus_men_pp"])] if shares else ["--"] * 3)
        robust.append(" & ".join(row) + r"\\")
    return ("\n".join(primary) + "\n", "\n".join(robust) + "\n")


def rating_table(ratings):
    body = [r"& & \multicolumn{3}{c}{All eligible} & \multicolumn{3}{c}{Image inventoried}\\",
            r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
            r"Minimum votes & Gender & Profiles & Mean & Pooled & Profiles & Mean & Pooled\\\midrule"]
    labels = {"votes_1": "1", "votes_20": "20", "votes_100": "100", "us_votes_20": "20, recorded U.S."}
    for key, label in labels.items():
        for sex, name, _, _ in SERIES:
            row = [label, name]
            for coverage in ("all", "inventoried"):
                cell = ratings[key]["all_ages"][sex][coverage]
                row.extend([f"{cell['n']:,}", format_rate(cell["mean_profile_like_rate"]),
                            format_rate(cell["vote_pooled_like_rate"])])
            body.append(" & ".join(row) + r"\\")
    return "\n".join(body) + "\n"


def height_synthesis_table(data):
    whyte_m = data["open_synthesis"]["whyte_2021"]["height_vs_attraction"]["Men"]
    pov_m = data["pov"]["height_percentiles"]["by_sex"]["Male"]
    lid = data["open_synthesis"]["lidborg_2022"]
    inch_map = {r["inches"]: r for r in pov_m["by_inch"]}

    lines = [
        r"Source \& male stature contrast & Sample ($n$) & Height pct. & Primary attraction or union metric & Secondary strategy or fertility metric\\\midrule",
        r"\multicolumn{5}{@{}l}{\emph{Panel A: Australian self-rated attraction \& partnering by male height bracket (Whyte et al.\ 2021, ages 18--64)}}\\",
    ]
    tot_m = sum(x["n"] for x in whyte_m["by_bracket"])
    cum_m = 0
    for b in whyte_m["by_bracket"]:
        lo_p = 100.0 * cum_m / tot_m
        cum_m += b["n"]
        hi_p = 100.0 * cum_m / tot_m
        if b["n"] >= 25:
            ci = b["attraction_percentile_ci_95"]
            lines.append(
                f"Ordinal band {b['height_code']} ({lo_p:.1f}--{hi_p:.1f}th pct.) & {b['n']:,} & {b['height_percentile']:.1f} & "
                f"Attraction {b['mean_attraction_percentile']:.1f}th pct.\\ $[{ci[0]:.1f}, {ci[1]:.1f}]$ & "
                f"Partnered {format_rate(b['partnered_share'])}; parent {format_rate(b['offspring_share'])}\\\\"
            )
    lines.append(r"\addlinespace\multicolumn{5}{@{}l}{\emph{Panel B: POV dating platform male stature distribution, 6'0'' rounding cliff \& prime-age outcomes (ages 30--49)}}\\")
    for inc in (61, 64, 67, 70, 71, 72, 73, 76, 78):
        r = inch_map[inc]
        out = r["outcomes_ages_30_49"]
        lbl = r["label"].replace('"', "''")
        lines.append(
            f"Reported {lbl} ({r['cm']}\\,cm; {r['observed_to_gaussian_ratio']:.2f}$\\times$ Gauss.) & {r['n_all_ages']:,} & {r['mean_height_percentile']:.1f} & "
            f"$>3$-yr union {format_rate(out['over_3y_union']['share'])}; athletic {format_rate(out['athletic_body']['share'])} & "
            f"Casual {format_rate(out['casual_intent']['share'])}; parent {format_rate(out['parenthood']['share'])}\\\\"
        )
    lines.append(r"\addlinespace\multicolumn{5}{@{}l}{\emph{Panel C: Cross-population meta-analysis of male stature vs.\ functional bodily strength (Lidborg et al.\ 2022)}}\\")
    for title, sub in (
        ("Male height $\\to$ mating outcomes", lid["height"]["mating"]),
        ("Male height $\\to$ reproduction / fertility", lid["height"]["reproduction_all"]),
        ("Male strength/muscularity $\\to$ mating", lid["body_strength"]["mating"]),
        ("Male strength/muscularity $\\to$ reproduction", lid["body_strength"]["reproduction_all"]),
    ):
        lo, hi = sub["re_ci_95"]
        lines.append(
            f"{title} ($k={sub['k']}$) & {sub['total_n']:,} & -- & "
            f"Random-effects $r = {sub['re_mean_r']:+.3f}$ $[{lo:+.3f}, {hi:+.3f}]$ & "
            f"Median $r = {sub['median_r']:+.3f}$; $N$-weighted $r = {sub['n_weighted_r']:+.3f}$\\\\"
        )
    return "\n".join(lines) + "\n"


def pov_geography_table(data):
    reps = data["pov"]["geography"]["replications"]
    totals = data["pov"]["geography"]["country_totals"]
    sfc = data["open_synthesis"]["okcupid_jse"]["sf_childless_fertility_crossover"]
    sf_w = sum(b["women"]["n"] for b in sfc["by_age_band"].values())
    sf_m = sum(b["men"]["n"] for b in sfc["by_age_band"].values())
    lines = [
        r"& \multicolumn{2}{c}{Eligible profiles} & \multicolumn{2}{c}{Childless ``Want children'' (W $-$ M, pp)} & \multicolumn{2}{c}{Serious commitment (W $-$ M, pp)}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"Country or open-science sub-cohort & Women & Men & Ages 25--29 & Ages 45--49 & Ages 25--29 & Ages 45--49\\\midrule",
    ]
    for cname in ("United States", "United Kingdom", "Canada", "Brazil", "Australia"):
        t = totals[cname]
        r = reps[cname]
        lines.append(
            f"{cname} & {t['Female']:,} & {t['Male']:,} & "
            f"{format_pp(r['25-29']['childless_want_women_minus_men_pp'])} & "
            f"{format_pp(r['45-49']['childless_want_women_minus_men_pp'])} & "
            f"{format_pp(r['25-29']['serious_women_minus_men_pp'])} & "
            f"{format_pp(r['45-49']['serious_women_minus_men_pp'])}\\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        f"San Francisco (\\texttt{{OkCupid-JSE}}, open; 45--54 right) & {sf_w:,} & {sf_m:,} & "
        f"{format_pp(sfc['by_age_band']['25-29']['women_minus_men_pp'])} & "
        f"{format_pp(sfc['by_age_band']['45-54']['women_minus_men_pp'])} & "
        f"\\multicolumn{{2}}{{c}}{{Crossover age: {sfc['interpolated_crossover_age']:.1f}\\,yr}}\\\\"
    )
    return "\n".join(lines) + "\n"


def benchmark_table(data):
    waves = {w["wave_id"]: w for w in data["inventory"]["multimodal_benchmark"]["waves"]}
    pf = data["facial_attribution"]["benchmark_10k_pure_face"]["test_overall"]
    lines = [r"Wave & Signal \& context representation ($N=10{,}000$ profiles; $20{,}000$ images) & Test Spearman $\rho$ & Test $\rho^2$ & Test beauty MAE\\\midrule"]
    selected = [
        ("H1", "Raw 1200px UV texture with unmasked 40\\% blue matte (ResNet-18 baseline)"),
        ("H3", "Foreground-masked UV crop removing background framing/matte artifact"),
        ("H4", "Domain human-face encoder (SFace 112px) + RegNetY / ResNet50-V2"),
        ("Pure-Face", "106-landmark Delaunay atlas + DINOv3-ViT-B/16 + ConvNeXtV2 + GlintR100 (zero metadata)"),
        ("H7", "Train-CDF percentile encoding + 5-objective homoscedastic head"),
        ("H10", "Cross-modal FiLM residual network + 5-fold grouped GBDT/CatBoost stacker"),
        ("H15", "Multi-photo candidate aggregation + pose/landmark geometry"),
        ("H18", "Cross-modal gated FiLM + low-rank bilinear ordinal neural ensemble"),
        ("H20", "Constrained exact-$L_1$ grand ensemble (DINOv3 + ConvNeXtV2 + GlintR100 + GBDT)"),
        ("H23", "Post-ensemble 3D mesh/TPS frontalization \\& Log-Gabor spectral relief (saturation check)"),
    ]
    for wid, desc in selected:
        if wid == "Pure-Face":
            r = pf["spearman_r"]
            lines.append(f"{wid} & {desc} & {r:.4f} & {r ** 2:.3f} & {pf['beauty_mae']:.4f}\\\\")
        else:
            pt = waves[wid]["profile_test"]
            r = pt["spearman_r"]
            lines.append(f"{wid} & {desc} & {r:.4f} & {r ** 2:.3f} & {pt['beauty_mae']:.4f}\\\\")
    return "\n".join(lines) + "\n"


def _tex_clean(text):
    rep = (
        ("&", r"\&"),
        ("%", r"\%"),
        ("\u2030", r"\text{\textperthousand}"),
        ("Ne\u2248", r"$N_e\approx$"),
        ("Coalescent Ne ", r"Coalescent $N_e$ "),
        ("CO2", r"$\mathrm{CO}_2$"),
        (" permil", r"\textperthousand{}"),
        ("\u2248", r"$\approx$"),
        ("\u2192", r"$\to$"),
        ("\u2014", "---"),
        ("\u2013", "--"),
        ("\u2212", "$-$"),
        ("\u00b1", r"$\pm$"),
        ("\u00d7", r"$\times$"),
        ("~", r"$\sim$"),
    )
    out = text
    for old, new in rep:
        out = out.replace(old, new)
    return out


def plateau_events_table(data):
    evs = data["paleoclimate_plateau"]["candidate_events"]
    cite_map = {
        "mpt_intensification": "lisiecki2005,clark2006,chalk2017,pena2014",
        "mis34_europe_cooling": "margari2023,muttoni2024",
        "bosumtwi_impact": "koeberl2007",
        "wonderwerk_fire": "berna2012",
        "fitcoal_bottleneck": "hu2023,zhou2026,deng2025,cousins2025,terhorst2025,ragsdale2023",
        "pantasma_impact": "rochette2019",
        "oldest_toba_tuff": "ito2020",
        "australasian_tektite_impact": "jourdan2019,sieh2020,sieh2024,mizera2024,michel2024",
        "gby_cooking": "zohar2022,wrangham2009",
        "matuyama_brunhes": "singer2019",
    }
    win_override = {
        "mpt_intensification": r"1.25--0.70\,Ma",
        "fitcoal_bottleneck": r"930--813\,ka",
        "pantasma_impact": r"815$\pm$11\,ka",
        "australasian_tektite_impact": r"789$\pm$3\,ka",
    }
    lines = []
    for ev in evs:
        eid = ev["id"]
        w_lo, w_hi = ev["window_ma"]
        if eid in win_override:
            win_str = win_override[eid]
        elif ev["point_ma"] >= 1.0:
            win_str = rf"$\sim${ev['point_ma']:.2f}\,Ma"
        else:
            win_str = rf"$\sim${1000.0 * ev['point_ma']:.0f}\,ka"
        start_ma = w_hi if (w_hi - w_lo) >= 0.11 else ev["point_ma"]
        lag_kyr = int(round(1000.0 * (1.50 - start_ma)))
        lag_str = rf"$+{lag_kyr}$\,kyr"
        cites = cite_map[eid]
        label_str = f"{_tex_clean(ev['short_label'])}~\\cite{{{cites}}}"
        tier_str = _tex_clean(ev["tier_label"])
        rec_str = _tex_clean(ev["evidence_summary"])
        scope_str = _tex_clean(ev["hominin_scope"])
        lines.append(f"{win_str} & {lag_str} & {label_str} & {tier_str} & {rec_str} & {scope_str}\\\\")
    return "\n".join(lines) + "\n"


def holoface_table(data):
    hb = data["holoface_benchmark"]
    waves = {w["wave_id"]: w for w in hb["waves_trajectory"]}
    hp = hb["holoface_pure_zero_metadata"]
    hf = hb["holoface_full_multimodal"]
    pc = hb["psychometric_voter_ceiling"]["tiers"]
    sh = {r["family_id"]: r for r in hp["shapley_4family_decomposition"]}
    fams = hp["families_standalone"]

    lines = [
        r"\multicolumn{8}{@{}l}{\emph{Panel A: 43-wave benchmark progression and \texttt{HoloFace-L1} synthesis ($N=1{,}000$ held-out test profiles, $n_{\text{votes}}\ge 20$)}}\\",
    ]
    wave_rows = [
        ("H1 Baseline", "Raw UV pixels", waves["H1"], r"Unmasked 40\% blue background matte confounds ResNet-18"),
        ("H4 Domain Face", "Aligned face pixels", waves["H4"], "SFace 112px + RegNetY/ResNet50-V2 human-face backbones"),
        ("H8 Pre-UV + Pose", "Visual + 72-pt mesh", waves["H8"], "Pre-UV landmark geometry without demographic metadata"),
    ]
    for lbl, reg, w, note in wave_rows:
        r = w["test_spearman_r"]
        lines.append(
            f"{lbl} & {reg} & {w['val_mae']:.4f} & {w['test_mae']:.4f} & {w['test_rmse']:.4f} & "
            f"{r:.3f} & {100.0 * (r ** 2):.1f}\\% & {note}\\\\"
        )
    lines.append(
        f"\\textbf{{HoloFace-Pure}} & Zero metadata (F1--F4) & {hp['val']['beauty_mae']:.4f} & "
        f"{hp['test']['beauty_mae']:.4f} & {hp['test']['beauty_rmse']:.4f} & "
        f"{hp['test']['spearman_r']:.3f} & {100.0 * hp['test']['spearman_r2']:.1f}\\% & "
        f"Strict zero-metadata visual convex synthesis ($\\rho={pc['votes_500_plus']['holoface_pure_spearman_r']:.3f}$ at 500+ votes)\\\\"
    )
    post_rows = [
        ("H10 FiLM + GBDT", "Multimodal", waves["H10"], "Cross-modal FiLM residual + 5-fold grouped GBDT/CatBoost"),
        ("H20 Exact-$L_1$ LP", "Multimodal", waves["H20"], "DINOv3 + ConvNeXtV2 + GlintR100 + CORAL/FiLM exact-$L_1$ LP"),
        ("H29 Kendall 3D", "Multimodal + 3D mesh", waves["H29"], r"Riemannian shape manifold ($\Sigma_2^{72}\times\Sigma_2^{106}\times\Sigma_3^{68}$); best single-wave RMSE"),
        ("H30 Cortical Router", "Multimodal + 12-space", waves["H30"], r"Retinotopic cortical fovea + dynamic 12-alignment router; best single-wave $\rho$"),
        ("H40 Set-Transformer", "Multimodal 20k views", waves["H40"], f"Permutation-invariant multi-photo set attention (single-image MAE {waves['H40']['single_image_test_mae']:.4f})"),
    ]
    for lbl, reg, w, note in post_rows:
        r = w["test_spearman_r"]
        lines.append(
            f"{lbl} & {reg} & {w['val_mae']:.4f} & {w['test_mae']:.4f} & {w['test_rmse']:.4f} & "
            f"{r:.3f} & {100.0 * (r ** 2):.1f}\\% & {note}\\\\"
        )
    lines.append(
        f"\\textbf{{HoloFace-Full ($L_1$)}} & 29-OOF exact-$L_1$ LP & {hf['val_l1']['beauty_mae']:.4f} & "
        f"\\textbf{{{hf['test_l1']['beauty_mae']:.4f}}} & {hf['test_l1']['beauty_rmse']:.4f} & "
        f"{hf['test_l1']['spearman_r']:.3f} & {100.0 * hf['test_l1']['spearman_r2']:.1f}\\% & "
        f"Best Val/Test/Single-Image MAE ({hf['single_image_test']['beauty_mae']:.4f}); gender Pct MAE {hf['test_l1']['gender_pct_mae']:.1f}\\\\"
    )
    lines.append(
        f"\\textbf{{HoloFace-Full ($L_2$)}} & Skew-unbiased ridge & -- & "
        f"{hf['test_l2_unbiased_head']['beauty_mae']:.4f} & \\textbf{{{hf['test_l2_unbiased_head']['beauty_rmse']:.4f}}} & "
        f"{hf['test_l2_unbiased_head']['spearman_r']:.3f} & {100.0 * hf['test_l2_unbiased_head']['spearman_r2']:.1f}\\% & "
        f"Eliminates male right-skew rating bias ($+0.088\\to {hf['subgroups_test_l2_gender']['Male']['bias']:+.3f}$)\\\\"
    )

    lines.append(r"\addlinespace\multicolumn{8}{@{}l}{\emph{Panel B: Standalone performance and exact $2^4=16$ Shapley share of zero-metadata \texttt{HoloFace-Pure} rank variance}}\\")
    fam_specs = [
        ("F4_foundation_12space", "F4 Foundation 12-space", "6,520-dim visual", "across 8 foundation/face backbones"),
        ("F1_geometry_3d", "F1 3D Kendall shape", "582-dim mesh/Jacobian", "pure 3D/2D landmarks, zero pixel intensities"),
        ("F2_skin_photometry", "F2 Skin chromophores", r"288-dim albedo/$1/f^\beta$", r"hemoglobin/melanin \& spectral micro-relief"),
        ("F3_staging_pose", r"F3 Staging \& pose", "106-dim pose/framing", r"changeable yaw/pitch, smile \& framing only"),
    ]
    for fid, flbl, freg, fdesc in fam_specs:
        fc = fams[fid]
        sc = sh[fid]
        lines.append(
            f"{flbl} & {freg} & {fc['val']['beauty_mae']:.4f} & {fc['test']['beauty_mae']:.4f} & {fc['test']['beauty_rmse']:.4f} & "
            f"{fc['test']['spearman_r']:.3f} & {100.0 * fc['test']['spearman_r2']:.1f}\\% & "
            f"Shapley share \\textbf{{{sc['shapley_share_of_pure_face_pct']:.1f}\\%}} ($+\\Delta\\rho^2={100.0 * sc['shapley_test_spearman_r2']:.1f}\\%$, {fdesc})\\\\"
        )

    lines.append(r"\addlinespace\multicolumn{8}{@{}l}{\emph{Panel C: Finite-voter psychometric Oracle ceiling ($\hat\sigma_{\text{voter}}^2=" + f"{hb['psychometric_voter_ceiling']['variance_components_train_fit']['pooled_train_oof_fit']['sigma_voter_var']:.2f}" + r"$) across held-out vote-depth tiers}}\\")
    tier_specs = [
        ("votes_20_49", "Oracle: 20--49 votes"),
        ("votes_50_199", "Oracle: 50--199 votes"),
        ("votes_200_plus", "Oracle: 200+ votes"),
        ("votes_500_plus", "Oracle: 500+ votes"),
        ("all_test", "Oracle: All test"),
    ]
    for tid, tlbl in tier_specs:
        tc = pc[tid]
        lines.append(
            f"{tlbl} ($n={tc['n']:,}$) & Median {tc['median_votes']:,.0f} raters & -- & "
            f"{tc['oracle_finite_vote_mae']:.4f} & -- & {tc['oracle_finite_vote_spearman_r']:.3f} & "
            f"{100.0 * tc['oracle_finite_vote_spearman_r2']:.1f}\\% & "
            f"\\texttt{{HoloFace-Full}} $\\rho={tc['holoface_full_spearman_r']:.3f}$ ($\\rho^2={100.0 * tc['holoface_full_spearman_r2']:.1f}\\%$); "
            f"irreducible single-context gap {100.0 * tc['irreducible_single_context_gap_full']:.1f}\\%\\\\"
        )
    return "\n".join(lines) + "\n"


def references_bbl(generated_tex=None):
    generated_tex = generated_tex or {}
    v9_raw = (V9_ROOT / "paper/main.bbl").read_text()
    first_bib = v9_raw.find(r"\bibitem")
    v9_body = v9_raw[first_bib:].replace(r"\end{thebibliography}", "") if first_bib >= 0 else ""

    bib_map = {}
    bib_re = re.compile(r"(\\bibitem(?:\[.*?\])?\{([^}]+)\}.*?)(?=\\bibitem(?:\[|\{)|\Z)", flags=re.DOTALL)
    for block in (v9_body, EXTRA_BIBITEMS.strip()):
        for m in bib_re.finditer(block):
            entry, key = m.group(1).strip(), m.group(2).strip()
            bib_map[key] = entry

    sections_dir = ROOT / "docs/sections"

    def _expand_tex(tex_str, seen_inputs=None):
        seen_inputs = set() if seen_inputs is None else set(seen_inputs)
        no_comments = re.sub(r"(?<!\\)%.*", "", tex_str)

        def _repl(match):
            inc = match.group(1).strip()
            if not inc.endswith(".tex"):
                inc_tex = inc + ".tex"
            else:
                inc_tex = inc
            if inc_tex in seen_inputs or inc_tex == "performance-references.tex":
                return ""
            next_seen = seen_inputs | {inc_tex}
            if inc_tex in generated_tex:
                return _expand_tex(generated_tex[inc_tex], next_seen)
            sec_path = sections_dir / inc_tex
            if sec_path.exists():
                return _expand_tex(sec_path.read_text(), next_seen)
            return ""

        return re.sub(r"\\(?:input|tabinput)\{([^}]+)\}", _repl, no_comments)

    full_doc = _expand_tex((sections_dir / "performance-before-outcomes.tex").read_text())
    cite_re = re.compile(r"\\cite[a-zA-Z*]*\s*(?:\[[^\]]*\]\s*)*\{([^}]+)\}")
    ordered_keys = []
    seen_keys = set()
    for m in cite_re.finditer(full_doc):
        for raw_k in m.group(1).split(","):
            k = raw_k.strip()
            if k and k not in seen_keys:
                seen_keys.add(k)
                ordered_keys.append(k)

    missing = [k for k in ordered_keys if k not in bib_map]
    assert not missing, f"Missing bibliography keys in references_bbl: {missing}"

    out_lines = [
        f"\\begin{{thebibliography}}{{{max(len(ordered_keys), 99)}}}",
        r"\providecommand{\natexlab}[1]{#1}",
        r"\providecommand{\url}[1]{\texttt{#1}}",
    ]
    for k in ordered_keys:
        out_lines.append(bib_map[k])
        out_lines.append("")
    out_lines.append(r"\end{thebibliography}")
    return "\n".join(out_lines) + "\n"


def primary_secondary_table(data):
    speed = data["speed_dating"]
    pov = data["pov"]
    osyn = data["open_synthesis"]
    kow = osyn["kowal_2025"]
    hcm = osyn["hcmst_2017"]
    lid = osyn["lidborg_2022"]
    joel = osyn["joel_2020"]
    sccs = osyn["betzig_sccs"]
    gss_all = osyn["v9_anchors"]["gss_three_episodes_all"]
    e17 = _v9_json("e17_speed")

    chk_k = kow["checklist_and_surplus_decomposition"]["pooled"]["predictors_sd_per_sd"]
    hvk = kow["homophily_vs_kindness"]
    rsk = kow["reverence_vs_settling_kink"]["by_sex"]
    hw = speed["checklist_and_hindsight"]["hobby_checklist_vs_live_wavelength"]
    sat_b = speed["checklist_and_hindsight"]["objective_sat_blindness"]
    mdv = osyn["okcupid_jse"]["ten_prompt_portfolio"]["multi_domain_lexical_validity"]
    ost = osyn["okcupid_jse"]["orientation_staging_experiment"]

    lines = [
        r"\multicolumn{5}{@{}l}{\emph{Panel A: Secondary Mating Criteria (high stranger legibility $\rho_X$, strong entry/sorting, near-zero bond or fertility welfare $\alpha_X \approx 0$)}}\\",
        rf"Single-photo facial rank & 6.68M HON; 20k \texttt{{HoloFace-L1}} & $\rho={data['holoface_benchmark']['holoface_full_multimodal']['test_l1']['spearman_r']:.3f}$ (27.3\% 500+ gap) & Entry $+{kow['partnering_slopes']['by_sex']['men']['looks']['pp_per_sd']:.1f}$/+${kow['partnering_slopes']['by_sex']['women']['looks']['pp_per_sd']:.1f}$\,pp/SD & Bond $+{chk_k['partner_looks']['estimate']:.2f}$\,SD ($\approx\frac{{1}}{{2}}$ kindness); aging divorce {100*sccs['causes']['old_age_loss_of_looks']['share_mentioned']:.1f}\%\\",
        rf"Male stature \& $6'0''$ threshold & 781.7k POV; 59.9k \texttt{{OkCupid}}; $k=97$ & Instant box; $+{ost['groups']['men_straight']['stature']['jump_5ft11_to_6ft0_pct']:.1f}\%$ straight surge & Mating $r={lid['height']['mating']['re_mean_r']:+.3f}$ & $>3$-yr union flat (85.6\%); fertility $r={lid['height']['reproduction_all']['re_mean_r']:+.3f}$ (null)\\",
        rf"Income \& ``marrying-up'' surplus & 90.5k Kowal; 34.1k U.S.\ GSS & Instant bracket; $1.66\times$ straight M show & Entry $+{kow['partnering_slopes']['by_sex']['men']['resources']['pp_per_sd']:.1f}$\,pp/SD; GSS $+{gss_all['income']['men']['pairing'][0]:.3f}$ & Bond ${chk_k['partner_resources']['estimate']:+.2f}$\,SD; marry-up ${chk_k['marrying_up_surplus']['estimate']:+.2f}$\,SD; marital births ${gss_all['income']['men']['number'][0]:+.3f}$\\",
        rf"17-hobby \& 5-trait checklist tags & 7,941 speed-date dyads; 47.9k couples & Instant profile tags & 17-hobby ``Yes'' $r={hw['r_objective_hobby_corr_yes']['estimate']:+.3f}$ (null) & 5-trait checklist fit $+{chk_k['idiographic_checklist_fit']['estimate']:.2f}$\,SD ($5.3\times$ below kindness)\\",
        rf"Politics, class \& degree sorting & 47.4k--50.0k couples (150 countries) & Instant credential / label & Assortative $r=+{hvk['politics']['assortative_pearson_r']:.2f}$ / $+{hvk['social_class']['assortative_pearson_r']:.2f}$ / $+{hvk['education']['assortative_pearson_r']:.2f}$ & Bond $+{hvk['politics']['homophily_prqc_sd_per_sd']['estimate']:.2f}$ / $+{hvk['social_class']['homophily_prqc_sd_per_sd']['estimate']:.2f}$ / $+{hvk['education']['homophily_prqc_sd_per_sd']['estimate']:.2f}$\,SD ($5.1\text{{--}}8.9\times$ below kindness)\\",
        r"\addlinespace\multicolumn{5}{@{}l}{\emph{Panel B: Primary Mating Criteria (low zero-context legibility $\rho_{K,0}\ll \rho_X$, multi-context witness audit, dominant love \& survival $\alpha_K \gg |\alpha_X|$)}}\\",
        rf"Everyday kindness \& warmth & 47.9k couples; 1.05M POV essays & 4-min $r_1={e17['jury']['sinc']['r1']:.2f}$ (near-blind) & Entry ${kow['partnering_slopes']['by_sex']['men']['kindness']['pp_per_sd']:+.2f}$/${kow['partnering_slopes']['by_sex']['women']['kindness']['pp_per_sd']:+.2f}$\,pp/SD ($\approx 0$) & \textbf{{$\#1$ bond driver ($+{chk_k['partner_kindness']['estimate']:.2f}$\,SD)}}; settling kink ${rsk['women']['country_fe_spline_sd_per_pt']['settling_deficit_kindness']['estimate']:+.2f}$\,SD/pt\\",
        rf"Live play, humor \& dyadic spark & 8,378 speed dates; 47.3k \texttt{{OkCupid}} & 60.5\% unique dyadic SRM variance & Live wavelength $r={hw['r_live_shared_wavelength_yes']['estimate']:+.3f}$ ($20\times$ tags) & $\#1$ ``Message me if'' cue ($25.4\times$ tall, $74.9\times$ wealth)\\",
        rf"General intelligence ($g$) \& curiosity & SAT dates ($n=6{{,}}822$); 9-domain essays & 1-date SAT $R^2={100*sat_b['stranger_intel_sat_r2']:.2f}\%$ (blind) & $K=9$ domains: $5.1\times$ lexical validity; GSS $+{gss_all['vocabulary']['men']['pairing'][0]:.3f}$ & Couple SAT $r={sat_b['r_matched_couple_sat']:+.2f}$; hindsight snaps back $+{speed['checklist_and_hindsight']['within_person_shifts']['threeweek_stated_minus_nextday_realized']['women']['delta_points']['intel']['estimate']:.1f}$\,pts\\",
        rf"Unstaged devotion \& witness patience & 43 longit.\ studies; 3,510 \texttt{{HCMST}} & App pre-romance $0.27$\,yr ($\div 9.7$ kin) & $\le 3$\,mo rush = \textbf{{$3.1\times$ breakup}} vs.\ $2.5+$ yr & Dyadic devotion explains \textbf{{{joel['satisfaction']['models']['ActorRelBaseline']['unweighted_mean_r2_pct']:.1f}\%}} vs.\ traits {joel['satisfaction']['models']['PartnerTraitBaseline']['unweighted_mean_r2_pct']:.1f}\% $R^2$\\",
        rf"Emotional fidelity \& hearth care & 160 SCCS soc.; 136 childcare soc. & Lullaby $5.0\times$ love-song $d$ (29.4k raters) & Hearth cooking verbs rise \textbf{{$4.2\text{{--}}4.9\times$}} (18--79) & Betrayal cited $14.3\times$ aging looks; $+{osyn['childhood_coevolution']['proposition_22_e22_coevolution']['love_devotion_gain_cm3']:.0f}\,\mathrm{{cm^3}}$ love gain\\",
    ]
    return "\n".join(lines) + "\n"


def macro_kpi_table(data):
    pp = data["paleoclimate_plateau"]
    pvc_500 = data["holoface_benchmark"]["psychometric_voter_ceiling"]["tiers"]["votes_500_plus"]
    speed = data["speed_dating"]
    dc = speed["dyadic_chemistry"]
    hw = speed["checklist_and_hindsight"]["hobby_checklist_vs_live_wavelength"]
    csm = speed["checklist_and_hindsight"]["cross_sex_misperception"]
    pov = data["pov"]
    swv = pov["self_written_voice"]
    sar = pov["second_act_reentry"]["headline_contrasts"]
    osyn = data["open_synthesis"]
    cc = osyn["childhood_coevolution"]
    nhs = osyn["nhs_campfire"]
    hcm = osyn["hcmst_2017"]
    kow = osyn["kowal_2025"]
    chk_k = kow["checklist_and_surplus_decomposition"]["pooled"]
    hvk = kow["homophily_vs_kindness"]
    joel = osyn["joel_2020"]
    sccs = osyn["betzig_sccs"]
    lid = osyn["lidborg_2022"]
    v9 = osyn["v9_anchors"]
    gss_all = v9["gss_three_episodes_all"]
    gss_m = gss_all["income"]["men"]
    gss_v = gss_all["vocabulary"]["men"]

    rows = [
        (
            r"\textbf{1. When Brains Grew vs.\ Stalled}",
            r"When did hominin brains expand, why did they stall for 800\,kyr (1.50--0.70\,Ma), and what released the surge?",
            f"{pp['plateau_summary']['gingerich_n']} crania + \\texttt{{LR04}} + {cc['sccs_alloparenting_and_sharing']['early_childhood_nonmaternal_majority_sccs52']['coded_societies']} SCCS societies",
            rf"Grew to $\sim\!908\,\mathrm{{cm^3}}$ by 1.5\,Ma, stalled \textbf{{800\,kyr}} (\textbf{{{pp['timing_gap_summary']['onset_to_mis34_cooling_kyr']:.0f}--{pp['timing_gap_summary']['onset_to_bolaven_impact_kyr']:.0f}\,kyr}} before crises), then surged \textbf{{$+50\%/\mathrm{{Myr}}$}} post-cooking; age-4.5 brain uses \textbf{{{cc['kuzawa_2014_brain_metabolism']['peak_rmr_pct']:.1f}\% RMR}}",
            rf"SCCS solo-mother care is \textbf{{0/136 (0.0\%)}}; cooking w/o pair-bond love stalls at $\sim\!{cc['proposition_22_e22_coevolution']['regimes']['cooking_without_pair_bond_love']['equilibrium_cm3_closed_form']:,.0f}\,\mathrm{{cm^3}}$ vs.\ {cc['proposition_22_e22_coevolution']['regimes']['late_homo_cooking_plus_pair_bond_love']['equilibrium_cm3_closed_form']:,.0f} (Props.~\ref{{prop:tissue}},~\ref{{prop:childhood}})",
        ),
        (
            r"\textbf{2. Multi-Modal Legibility Wall}",
            "Why do zero-context strangers over-weight secondary cues while missing primary mind and devotion?",
            r"20,000 \texttt{HoloFace-L1} portraits + 8,378 dates + 484k essays + 118 songs",
            rf"\textbf{{{100.0 * pvc_500['irreducible_single_context_gap_full']:.1f}\%}} photo rank-variance gap; 4-min stranger explains \textbf{{{100.0 * speed['checklist_and_hindsight']['objective_sat_blindness']['stranger_intel_sat_r2']:.2f}\%}} of partner SAT vs.\ \textbf{{{osyn['okcupid_jse']['ten_prompt_portfolio']['multi_domain_lexical_validity']['k9_to_k1_validity_ratio']:.1f}$\times$}} across 9 essay domains",
            rf"Blind raters ($N\!=\!29{{,}}357$) decode lullabies ($d\!=\!+{nhs['acoustic_legibility_gradient']['lullaby']['cohen_d']:.2f}$) \textbf{{{nhs['lullaby_to_love_d_ratio']:.1f}$\times$}} over love songs ($+{nhs['acoustic_legibility_gradient']['love']['cohen_d']:.2f}$; Props.~\ref{{prop:portfolio}},~\ref{{prop:twoway}})",
        ),
        (
            r"\textbf{3. Mind-Reading Error \& Play}",
            "How badly do men and women misread opposite-sex priorities, and what creates live romantic spark?",
            rf"{dc['variance_partition']['like']['n']:,} ratings + {hw['n_common_dyad_decisions']:,} hobby dyads + 690 matches",
            rf"Both sexes overestimate opposite-sex looks weight (\textbf{{$+{csm['women_about_men']['delta_points']['attr']['estimate']:.1f}$ W / $+{csm['men_about_women']['delta_points']['attr']['estimate']:.1f}$ M pts}}) \& underestimate intelligence (\textbf{{${csm['women_about_men']['delta_points']['intel']['estimate']:+.1f}$ / ${csm['men_about_women']['delta_points']['intel']['estimate']:+.1f}$}}); men call \textbf{{{speed['checklist_and_hindsight']['post_event_followup_3_4_weeks']['initiation_by_sex']['male_to_female_calls_per_match_ratio']:.1f}$\times$}} more",
            rf"17-hobby overlap is \textbf{{null ($r\!=\!{hw['r_objective_hobby_corr_yes']['estimate']:+.3f}$)}} vs.\ live play \textbf{{$r\!=\!{hw['r_live_shared_wavelength_yes']['estimate']:+.3f}$ ({hw['live_to_checklist_r_ratio']:.1f}$\times$)}}; \textbf{{{100.0 * dc['variance_partition']['like']['unbiased_srm_shares']['unique_dyadic_and_noise']:.1f}\%}} unique SRM spark (Prop.~\ref{{prop:love}}(ii))",
        ),
        (
            r"\textbf{4. Voice vs.\ Secondary Staging}",
            r"What do men and women seek in their own words vs.\ how secondary platform filters trigger strategic staging?",
            rf"{swv['eligible_with_essay']['Pooled']:,} POV + 484k \texttt{{OkCupid}} essays + 65.9k effort logs",
            rf"Women write family (\textbf{{{swv['voice_to_checkbox_ratios']['female_family_to_tall']:.0f}$\times$}} $6'0''$); ``message me if'' asks humor \textbf{{{osyn['okcupid_jse']['ten_prompt_portfolio']['essay9_message_me_if']['ratios']['humor_to_tall']:.1f}$\times$}} tall \& \textbf{{{osyn['okcupid_jse']['ten_prompt_portfolio']['essay9_message_me_if']['ratios']['humor_to_wealth']:.1f}$\times$}} wealth; courtship effort drops \textbf{{$-14\%$/$-20\%$}} in marriage",
            rf"Straight men surge \textbf{{$+{osyn['okcupid_jse']['orientation_staging_experiment']['groups']['men_straight']['stature']['jump_5ft11_to_6ft0_pct']:.1f}\%$}} at $6'0''$ (gay men \textbf{{${osyn['okcupid_jse']['orientation_staging_experiment']['groups']['men_gay']['stature']['jump_5ft11_to_6ft0_pct']:+.1f}\%$}}) \& show income \textbf{{{osyn['okcupid_jse']['orientation_staging_experiment']['straight_to_gay_male_income_disclosure_ratio']:.2f}$\times$}} more (Props.~\ref{{prop:staging}},~\ref{{prop:lifecycle}}(i))",
        ),
        (
            r"\textbf{5. Audit Collapse \& Rush Hazard}",
            "What happens when digital apps compress pre-romantic witness audits and couples rush into lock-in?",
            rf"{hcm['total_respondents']:,} Stanford \texttt{{HCMST}} adults ({hcm['cohabitation_rush_hazard']['n_ever_cohabited']:,} cohabited)",
            rf"Apps compress pre-romance acquaintance \textbf{{{hcm['audit_compression_ratios']['institutional_kin_to_app_mean_ratio']:.1f}$\times$}} ($0.27$ vs.\ $2.58$\,yr); rushing to cohabit in \textbf{{$\le 3$\,mo triples breakup}} ({100*hcm['cohabitation_rush_hazard']['by_romance_to_cohab_window']['le_3m_rush']['breakup_share']:.1f}\% vs.\ {100*hcm['cohabitation_rush_hazard']['by_romance_to_cohab_window']['gt_2_5y']['breakup_share']:.1f}\%)",
            rf"Non-marital breakups are near-symmetric, whereas \textbf{{marital divorces flip to {100*hcm['breakup_initiation_asymmetry']['married_divorce_or_separation']['woman_share_of_unilateral']:.1f}\% female initiation}} (85.7\% in husbands' reports; Prop.~\ref{{prop:lifecycle}}(ii--iii))",
        ),
        (
            r"\textbf{6. Primary Love vs.\ Homophily}",
            "Do secondary sorting tags (politics, class, degrees, money) or primary kindness and reverence sustain love?",
            rf"{kow['inclusion_flow']['source_rows']:,} adults ({kow['checklist_and_surplus_decomposition']['n_total']:,} couples) + 43 longit.\ studies",
            rf"Primary kindness (\textbf{{${chk_k['predictors_sd_per_sd']['partner_kindness']['estimate']:+.2f}$\,SD}}) beats politics/class/degree homophily (\textbf{{{hvk['politics']['kindness_to_homophily_ratio']:.1f}--{hvk['education']['kindness_to_homophily_ratio']:.1f}$\times$}}), checklist fit (\textbf{{{chk_k['kindness_to_checklist_ratio']:.1f}$\times$}}), money, \& marry-up ($\approx 0$)",
            rf"Settling on kindness penalizes love \textbf{{1.8$\times$}} steeper than surplus adds; 43 longit.\ studies: devotion \textbf{{{joel['satisfaction']['models']['ActorRelBaseline']['unweighted_mean_r2_pct']:.1f}\%}} vs.\ traits {joel['satisfaction']['models']['PartnerTraitBaseline']['unweighted_mean_r2_pct']:.1f}\% $R^2$ (Prop.~\ref{{prop:love}})",
        ),
        (
            r"\textbf{7. Clocks, Verbs \& Second Acts}",
            "How do male and female desires, hearth verbs, and remarriage choices diverge or converge from 18 to 79?",
            r"1.14M POV + 1.05M essays + 85.0k jealousy + 160 SCCS",
            rf"Childless ``Want kids'' flips at \textbf{{Age $\sim\!{pov['childless_crossover']['band_midpoint_interpolated_age']:.0f}$}}, casual SOI widens ($1.46\to 1.77\times$ M$>$W), yet hearth cooking verbs rise \textbf{{$4.2\text{{--}}4.9\times$}} \& commitment stays W$>$M",
            rf"40--49 divorced W cut marriage intent \textbf{{$-{abs(sar['divorced_vs_single_female_marry_change_pct_40_49']):.0f}\%$}} (keeping \textbf{{80.8\%}} serious bond), while widowed M exceed W (Prop.~\ref{{prop:lifecycle}}(iv))",
        ),
        (
            r"\textbf{8. The 94\% Modern Pairing Gate}",
            "Did modern births fall because adults stopped wanting families or because stranger markets stall union entry?",
            r"34,092 U.S.\ GSS adults (incl.\ \texttt{WORDSUM}) + 232 meta-effects",
            rf"\textbf{{94\%}} of U.S.\ birth drop sits in non-marriage; male pairing is positive on income (${gss_m['pairing'][0]:+.3f}$), schooling ($+{gss_all['schooling']['men']['pairing'][0]:.3f}$), \& \texttt{{WORDSUM}} (\textbf{{${gss_v['pairing'][0]:+.3f}$}})",
            rf"Within-marriage births invert all four (${gss_m['number'][0]:+.3f}$ to ${gss_all['schooling']['men']['number'][0]:+.3f}$); height fertility null ($r\!=\!{lid['height']['reproduction_all']['re_mean_r']:+.3f}$; Props.~\ref{{prop:episodes}},~\ref{{prop:waiting}})",
        ),
    ]
    return "\n".join(" & ".join(r) + r"\\" for r in rows) + "\n"


def generate(data):
    pov, hon, speed = data["pov"], data["hon"], data["speed_dating"]
    responses, sensitivity = pov_tables(pov)
    plateau_tbl = plateau_events_table(data)
    holo_tbl = holoface_table(data)
    macro_tbl = macro_kpi_table(data)
    prim_sec_tbl = primary_secondary_table(data)
    gen_tex = {
        "performance-macro-kpi-table.tex": macro_tbl,
        "performance-primary-secondary-table.tex": prim_sec_tbl,
        "performance-pov-responses.tex": responses,
        "performance-pov-sensitivity.tex": sensitivity,
        "performance-hon-sensitivity.tex": rating_table(hon["ratings"]),
        "performance-height-table.tex": height_synthesis_table(data),
        "performance-pov-geography.tex": pov_geography_table(data),
        "performance-benchmark-table.tex": benchmark_table(data),
        "performance-face-drivers-table.tex": face_drivers_table(data),
        "performance-plateau-events-table.tex": plateau_tbl,
        "performance-holoface-table.tex": holo_tbl,
    }
    out = {
        "performance-numbers.tex": numbers(data),
        "performance-extra-bib.tex": EXTRA_BIBITEMS,
        **gen_tex,
        "performance-references.tex": references_bbl(gen_tex),
        "performance-fig1-main.pdf": fig1_main_plot(data),
        "performance-fig-fossils.pdf": fossils_paleoclimate_plot(data),
        "performance-fig-jury-choice.pdf": jury_choice_plot(data),
        "performance-fig-generalist-verify.pdf": generalist_verify_plot(data),
        "performance-fig-height-attraction.pdf": height_attraction_plot(data),
        "performance-fig-ethnography.pdf": ethnography_ancestors_plot(data),
        "performance-fig-world-synthesis.pdf": world_synthesis_plot(data),
        "performance-fig-love-bond.pdf": love_bond_plot(data),
        "performance-fig-primary-secondary-dynamics.pdf": primary_secondary_dynamics_plot(data),
        "performance-fig-hon-architecture.pdf": hon_architecture_plot(data),
        "performance-fig-holoface-ceiling.pdf": holoface_ceiling_plot(data),
        "performance-fig-face-drivers.pdf": face_drivers_plot(data),
        "performance-fig-face-age.pdf": face_age_plot(data),
        "performance-fig-face-world.pdf": face_world_plot(data),
        "performance-fig-pov-lifecourse.pdf": pov_lifecourse_plot(data),
        "performance-fig-gss-fertility.pdf": gss_fertility_gate_plot(data),
        "performance-fig-ai-benchmark.pdf": ai_benchmark_plot(data),
        "performance-parenthood.pdf": pov_plot(pov, data["cohorts"]["pov_bands"]),
        "performance-beauty.pdf": beauty_plot(hon["beauty_by_age"]),
        "performance-choice.pdf": choice_plot(speed),
        "performance-ratings.pdf": rating_plot(hon["ratings"]["votes_20"]),
    }
    return out
