#!/usr/bin/env bash
# Download every raw source used by the pipeline into data/raw/.  Idempotent: existing files are kept.
#
#  Greek
#   codex-sinaiticus/   Codex Sinaiticus TEI transcription, ITSEE Birmingham           CC BY-NC-SA 3.0
#   vaticanus/          Codex Vaticanus NT, CNTR transcription (MES format)             CC BY-SA 4.0
#   vaticanus/swete/    Vaticanus OT = Swete's Old Testament in Greek (nathans/lxx-swete,
#                       rebased on First1KGreek)                                        CC BY-SA 4.0
#   first1kgreek/       Acts of John, Thomas, Philip, Barnabas (Bonnet)                  CC BY-SA 4.0
#   apostolic-fathers/  Lake's Apostolic Fathers, corrected (Tauber & Macdonald)         CC BY-SA 4.0
#  Hebrew
#   oshb/               Leningrad Codex, Westminster/OSHB OSIS XML                      text PD, morph CC BY 4.0
#   mam/                Aleppo Codex text: Miqra according to the Masorah (OSIS)         CC BY-SA 4.0
#   samaritan/          Samaritan Pentateuch, DT-UCPH Text-Fabric (Schorch)              CC BY-NC 4.0
#   dss/                Dead Sea Scrolls incl. 1QIsaa, ETCBC Text-Fabric (Abegg)        CC BY-NC 4.0
#   inscriptions/       Nash Papyrus, Ketef Hinnom amulets (readings via Wikipedia)     CC BY-SA 4.0
#  Arabic
#   quran/              Quran, Tanzil Uthmani text + sura metadata                       Tanzil terms (verbatim, attribution)
#   bukhari/            Sahih al-Bukhari, Arabic, with book divisions (hadith-api)      Unlicense (public domain)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"
fetch() {  # fetch <url> <dest>
  if [ ! -s "$2" ]; then
    mkdir -p "$(dirname "$2")"
    curl -sSL --fail -o "$2" "$1" && echo "downloaded $(basename "$2")"
  fi
}

# --- Greek ------------------------------------------------------------------------------------------
[ -d "$RAW/codex-sinaiticus" ] || git clone --depth 1 https://github.com/itsee-birmingham/codex-sinaiticus.git "$RAW/codex-sinaiticus"
[ -d "$RAW/apostolic-fathers" ] || git clone --depth 1 https://github.com/jtauber/apostolic-fathers.git "$RAW/apostolic-fathers"
F1K="https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master/data"
for author in tlg0317 tlg2038 tlg2948 tlg2949; do
  fetch "$F1K/$author/tlg001/$author.tlg001.1st1K-grc1.xml" "$RAW/first1kgreek/$author.tlg001.1st1K-grc1.xml"
done
# CNTR: every NT witness up to c. AD 400 (papyri and majuscules incl. Vaticanus 03, Alexandrinus 02)
if [ -z "$(ls "$RAW/cntr/class1"/*.txt 2>/dev/null)" ]; then
  mkdir -p "$RAW/cntr/class1"
  tmp="$(mktemp -d)"
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/Center-for-New-Testament-Restoration/transcriptions.git "$tmp/cntr"
  (cd "$tmp/cntr" && git sparse-checkout set "class 1" >/dev/null)
  cp "$tmp/cntr/class 1/"*.txt "$RAW/cntr/class1/" && rm -rf "$tmp" && echo "downloaded CNTR class 1 ($(ls "$RAW/cntr/class1"/*.txt | wc -l) witnesses)"
fi
# Swete: one file per book; the repo is small, so a shallow clone avoids the rate-limited GitHub API.
SW="$RAW/vaticanus/swete"
if [ ! -d "$SW" ] || [ -z "$(ls "$SW"/*.txt 2>/dev/null)" ]; then
  mkdir -p "$SW"
  tmp="$(mktemp -d)"
  git clone --depth 1 -q https://github.com/nathans/lxx-swete.git "$tmp/lxx-swete"
  cp "$tmp"/lxx-swete/data/*.txt "$SW"/ && rm -rf "$tmp" && echo "downloaded Swete ($(ls "$SW"/*.txt | wc -l) books)"
fi

# --- Hebrew -----------------------------------------------------------------------------------------
OSHB="https://raw.githubusercontent.com/openscriptures/morphhb/master/wlc"
for b in Gen Exod Lev Num Deut Josh Judg Ruth 1Sam 2Sam 1Kgs 2Kgs 1Chr 2Chr Ezra Neh Esth Job Ps Prov Eccl Song \
         Isa Jer Lam Ezek Dan Hos Joel Amos Obad Jonah Mic Nah Hab Zeph Hag Zech Mal; do
  fetch "$OSHB/$b.xml" "$RAW/oshb/$b.xml"
done
fetch "https://raw.githubusercontent.com/bdenckla/MAM-basics/main/MAM-OSIS/mapm.osis.xml" "$RAW/mam/mapm.osis.xml"
SP="https://raw.githubusercontent.com/DT-UCPH/sp/main/tf/7.1.3"
for f in otext otype oslots sign g_cons_utf8 book chapter verse trailer; do
  fetch "$SP/$f.tf" "$RAW/samaritan/tf/$f.tf"
done
if [ ! -f "$RAW/dss/tf/2.0.1/otype.tf" ]; then
  mkdir -p "$RAW/dss/tf/2.0.1"
  fetch "https://github.com/ETCBC/dss/releases/download/v2.0.1/tf-2.0.1.zip" "$RAW/dss/tf-2.0.1.zip"
  (cd "$RAW/dss/tf/2.0.1" && unzip -q -o ../../tf-2.0.1.zip)   # the zip is flat: 80 .tf files
  echo "unpacked ETCBC dss 2.0.1 ($(ls "$RAW/dss/tf/2.0.1"/*.tf | wc -l) features)"
fi

# --- Arabic -----------------------------------------------------------------------------------------
fetch "https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=txt-2&agree=true" "$RAW/quran/quran-uthmani.txt"
fetch "https://tanzil.net/res/text/metadata/quran-data.xml" "$RAW/quran/quran-data.xml"
# Sahih al-Bukhari: the Arabic edition carries `reference.book`, so each kitab becomes a work.
fetch "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/ara-bukhari.json" "$RAW/bukhari/ara-bukhari.json"

echo "sources ready under $RAW"
