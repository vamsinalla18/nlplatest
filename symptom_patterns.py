import re

# Canonical symptom name → list of regex patterns covering patient language.
# Imported by scripts/build_kg.py and inference/disease_predictor.py.

SYMPTOM_PATTERNS = {
    # ── Vital signs / systemic ──────────────────────────────────────────────
    "fever": [
        r"\bfever\b", r"\bfeverish\b", r"\bhigh temperature\b", r"\bhigh fever\b",
        r"\btemperature is (high|up|elevated)\b", r"\brunning (a )?fever\b",
        r"\bbody (is |feels? )?(hot|burning|warm)\b", r"\bfeeling (hot|warm)\b",
    ],
    "chills": [
        r"\bchill(s)?\b", r"\bshiver(ing)?\b", r"\bfeeling cold\b",
        r"\bshaking (with )?cold\b", r"\bcold (and shaking|shaking)\b",
    ],
    "fatigue": [
        r"\bfatigue\b", r"\btired\b", r"\bexhausted\b", r"\brun.?down\b",
        r"\bno energy\b", r"\bworn out\b", r"\balways tired\b", r"\btired all the time\b",
        r"\bfeeling (drained|exhausted|wiped out)\b", r"\bno strength\b",
        r"\blethargy\b", r"\blethargic\b", r"\bno stamina\b",
    ],
    "weakness": [
        r"\bweak(ness)?\b", r"\blimb weakness\b", r"\bmuscle weakness\b",
        r"\bfeel(ing)? weak\b", r"\bno strength\b", r"\bcan.?t\b lift\b",
        r"\barms?.{0,10}(weak|heavy)\b", r"\blegs?.{0,10}(weak|heavy|give(s)? (out|way))\b",
    ],
    "sweating": [
        r"\bsweat(ing)?\b", r"\bperspir(ation|ing)\b", r"\bnight sweat(s)?\b",
        r"\bdrenched in sweat\b", r"\bsoaking (wet|in sweat)\b",
        r"\bwoke up? (drenched|soaked|sweating)\b",
    ],
    "weight_loss": [
        r"\bweight loss\b", r"\blosing weight\b", r"\blost (a lot of )?weight\b",
        r"\bweight has (dropped|decreased|gone down)\b", r"\bunintentional weight loss\b",
        r"\bclothes (are |are )?too big\b",
    ],
    "weight_gain": [
        r"\bweight gain\b", r"\bgaining weight\b", r"\bgained weight\b",
        r"\bputting on weight\b", r"\bweight has (increased|gone up|going up)\b",
        r"\bcan.?t\b stop gaining weight\b", r"\bunexpected weight gain\b",
    ],

    # ── Head / Neuro ────────────────────────────────────────────────────────
    "headache": [
        r"\bheadaches?\b", r"\bhead (ache|pain|hurts?|pounding|throbbing|pressure)\b",
        r"\bpain in (my )?head\b", r"\bmigraine\b", r"\bmy head (is killing|hurts|aches)\b",
        r"\bpressure in (my )?head\b", r"\bbanging headache\b", r"\bsinus (headache|pressure)\b",
    ],
    "dizziness": [
        r"\bdizzy\b", r"\bdizziness\b", r"\blightheaded\b", r"\bvertigo\b",
        r"\boff.?balance\b", r"\bbalance (problem|issue|trouble|loss)\b",
        r"\blosing (my )?balance\b", r"\broom (is )?spinning\b",
        r"\bfeeling (faint|like (I might )?pass out|woozy)\b", r"\bgiddy\b",
    ],
    "confusion": [
        r"\bconfused?\b", r"\bconfusion\b", r"\bdisoriented?\b", r"\bdisorientation\b",
        r"\bcan.?t\b (think|concentrate) (clearly|properly|straight)\b",
        r"\bmentally (confused?|foggy|unclear)\b", r"\bdon.?t\b know where (I am|I'm)\b",
        r"\bnot making sense\b", r"\bbrain fog\b", r"\bmind is (foggy|cloudy|blank)\b",
        r"\bcan.?t\b remember (where|what|who)\b", r"\bfeeling (out of it|disoriented)\b",
    ],
    "numbness": [
        r"\bnumb(ness)?\b", r"\btingling\b", r"\bpins and needles\b",
        r"\bfeel(ing)? nothing\b", r"\bno sensation\b", r"\bcan.?t\b feel (my )?(arm|leg|hand|foot|feet|fingers?|toes?)\b",
        r"\barms?.{0,10}(numb|tingly|tingling)\b", r"\blegs?.{0,10}(numb|tingly)\b",
    ],
    "loss_of_balance": [
        r"\bloss of balance\b", r"\bcan.?t\b (keep|maintain) (my )?balance\b",
        r"\bunsteady\b", r"\bbalance problems?\b", r"\bkeep (falling|tripping)\b",
        r"\bcan.?t\b walk straight\b", r"\bfalling (over|down|a lot)\b",
        r"\bstaggering\b", r"\bwobbling?\b", r"\bcoordination (problems?|issues?)\b",
    ],
    "tremor": [
        r"\btremor(s)?\b", r"\bshak(y|ing) hands?\b", r"\bhands? (are )?shaking\b",
        r"\buncontrollable shaking\b", r"\bbody shaking\b", r"\btwitching\b",
        r"\bhand(s)? (tremble|trembling|tremor)\b", r"\bcan.?t\b hold (things|stuff|cup|glass)\b",
        r"\bshaking (uncontrollably|badly)\b", r"\bquivering\b",
    ],
    "insomnia": [
        r"\bcan.?t\b sleep\b", r"\binsomnia\b", r"\bsleepless(ness)?\b",
        r"\btrouble sleeping\b", r"\bdifficulty (falling |getting )?asleep\b",
        r"\bno sleep\b", r"\bawake all night\b",
        r"\bwake up (at night|frequently|often|multiple times)\b",
        r"\bsleep (problems?|issues?|trouble|disorder)\b", r"\bcan.?t\b get (any |enough )?sleep\b",
        r"\bstay(ing)? awake\b", r"\brest(less|lessness)\b",
    ],
    "sadness": [
        r"\bfeel(ing)? (sad|depressed|hopeless|empty|worthless|down)\b",
        r"\bdepression\b", r"\bdepressed\b", r"\bpersistent sadness\b",
        r"\blow mood\b", r"\bdon.?t\b enjoy (anything|life)\b", r"\blost interest\b",
        r"\bno joy\b", r"\bno motivation\b", r"\bwant to cry (all the time|constantly)\b",
        r"\bfeeling (miserable|unhappy|blue)\b", r"\bcrying (a lot|all the time)\b",
    ],
    "anxiety_feeling": [
        r"\bfeel(ing)? anxious\b", r"\banxiety\b", r"\bworried (all the time|constantly|too much|excessively)\b",
        r"\bexcessive worry\b", r"\bpanic\b", r"\bpanic attack(s)?\b",
        r"\bnervous(ness)?\b", r"\bconstant worry\b", r"\bcan.?t\b stop worrying\b",
        r"\bfeel(ing)? on edge\b", r"\brestless(ness)?\b", r"\bdread\b",
    ],

    # ── Eyes ────────────────────────────────────────────────────────────────
    "blurred_vision": [
        r"\bblurr(ed|y) vision\b", r"\bvision (is )?blurr\b",
        r"\bdistorted vision\b", r"\bvisual disturbance\b", r"\bvision problem(s)?\b",
        r"\bcan.?t\b see (clearly|properly|well)\b", r"\bthings (look|are) (blurry|fuzzy)\b",
        r"\beyes (are )?blurry\b",
    ],
    "vision_loss": [
        r"\bvision (loss|lost|gone|failing)\b", r"\bcan.?t\b see\b",
        r"\bblind(ness)?\b", r"\blost (my )?vision\b",
        r"\bsudden(ly)? (lost|losing) (my )?vision\b", r"\bcannot see\b",
        r"\bgoing blind\b", r"\bdark (in one|in my) eye\b",
    ],
    "watery_eyes": [
        r"\bwatery eyes\b", r"\beyes (are )?watery\b",
        r"\bitchy eyes\b", r"\beye itching\b", r"\btearing\b",
        r"\beyes (are )?tearing\b", r"\btears (running|streaming)\b",
    ],
    "eye_redness": [
        r"\bred eyes\b", r"\beyes (are )?red\b", r"\bbloodshot\b",
        r"\beye(s)? (irritat|inflam|pink)\b", r"\bpink eye\b", r"\bconjunctivitis\b",
    ],
    "photophobia": [
        r"\bsensitiv(e|ity) to light\b", r"\blight (sensitive|sensitivity)\b",
        r"\bbothered by light\b", r"\blight (hurts?|bothers?|burns?)\b",
        r"\bcan.?t\b stand (the )?light\b", r"\bbright (light|lights) (hurt|bother)\b",
    ],
    "retro_orbital_pain": [
        r"\bpain (behind|back of) (my )?eyes?\b", r"\beye(s)? (pain|ache)\b",
        r"\bback of (my )?(eye|eyes) hurt\b", r"\bbehind (my )?eyes? hurt\b",
        r"\beye (pain|aching|hurting)\b",
    ],

    # ── Ear ─────────────────────────────────────────────────────────────────
    "ear_pain": [
        r"\bear (pain|ache|hurts?|sore)\b", r"\bearache\b",
        r"\bpain in (my |an )?(ear|ears)\b", r"\bmy ear hurts?\b",
        r"\bear is (throbbing|pounding|killing me)\b",
    ],
    "ear_discharge": [
        r"\bear discharge\b", r"\bfluid (from|coming from|draining from|out of) (my )?(ear|ears)\b",
        r"\bear draining\b", r"\bseeping from (my )?ear\b",
        r"\bear (leaking|weeping|oozing)\b", r"\bstuff coming out of (my )?ear\b",
    ],

    # ── Face / Speech ───────────────────────────────────────────────────────
    "facial_drooping": [
        r"\bface (is )?drooping?\b", r"\bdrooping? (face|mouth|eyelid|eye)\b",
        r"\bone side of (my )?face\b", r"\bfacial (droop|weakness|numbness|paralysis)\b",
        r"\bmouth (is )?drooping?\b", r"\bcrooked (face|smile|mouth)\b",
        r"\bface (looks|is) (uneven|lopsided|asymmetric)\b",
    ],
    "slurred_speech": [
        r"\bslurred? speech\b", r"\bcan.?t\b (speak|talk) (clearly|properly|well|normally)\b",
        r"\bspeech (is |is )?slurred?\b", r"\bdifficulty speaking\b",
        r"\bhard to (speak|talk)\b", r"\bwords (are |coming out )?(slurred|jumbled|wrong|garbled)\b",
        r"\btrouble (speaking|talking|forming words)\b", r"\bspeech (problem|difficulty|issue)\b",
    ],

    # ── Nose / Throat / Upper Respiratory ───────────────────────────────────
    "runny_nose": [
        r"\brunny nose\b", r"\bnose (is )?running\b", r"\bnasal discharge\b",
        r"\bnasal drip\b", r"\bnose (dripping|leaking)\b", r"\bdripping nose\b",
    ],
    "nasal_congestion": [
        r"\bcongested\b", r"\bnasal congestion\b", r"\bstuffed.?up\b",
        r"\bstuffy nose\b", r"\bblocked nose\b", r"\bsinus (pressure|congestion|blocked)\b",
        r"\bcan.?t\b breathe through (my )?(nose|nostrils)\b", r"\bnose (is )?blocked\b",
    ],
    "sneezing": [
        r"\bsneezing?\b", r"\bsneezes?\b", r"\bcant stop sneezing\b",
        r"\bkeep sneezing\b",
    ],
    "sore_throat": [
        r"\bsore throat\b", r"\bthroat (is |is )?sore\b",
        r"\bthroat (hurts?|aches?|scratchy|raw|irritated)\b",
        r"\bscratchy throat\b", r"\bthroat pain\b", r"\bpainful (to |to )?swallow\b",
        r"\bhard to swallow\b", r"\bhurts? to swallow\b",
    ],
    "throat_swelling": [
        r"\bthroat (is )?swollen\b", r"\bswollen throat\b",
        r"\bdifficulty swallowing\b", r"\bhard to swallow\b",
        r"\btroubl(e|ed) swallowing\b", r"\bfood (gets )?stuck in (my )?throat\b",
        r"\bdysphagia\b", r"\bcan.?t\b swallow\b",
    ],
    "loss_of_smell": [
        r"\bcan.?t\b smell\b", r"\bloss of smell\b", r"\bno sense of smell\b",
        r"\bsmell (is |is )?gone\b", r"\banosmia\b",
        r"\bcan.?t\b (detect|notice) (any )?smell\b", r"\bsmells? (nothing|nothing at all)\b",
        r"\bsmell has (gone|disappeared|left)\b",
    ],
    "loss_of_taste": [
        r"\bcan.?t taste\b", r"\bloss of taste\b", r"\bno sense of taste\b",
        r"\btaste (is |is )?gone\b", r"\bageusia\b", r"\bfood has no taste\b",
        r"\beverything tastes? (bland|same|nothing|like nothing)\b",
        r"\btaste has (gone|disappeared|left)\b",
        r"\bno (taste|flavor)\b", r"\blost (my )?taste\b",
        r"\bcan.?t taste (anything|food|at all)\b",
        r"\b(smell (or|and) taste|taste (or|and) smell)\b",
    ],
    "mucus_production": [
        r"\bphlegm\b", r"\bmucus\b", r"\bsputum\b", r"\bproductive cough\b",
        r"\bcough(ing)? up (phlegm|mucus|sputum)\b", r"\bwet cough\b",
        r"\bchest congestion\b", r"\bcoughing up stuff\b",
    ],

    # ── Chest / Cardiac ─────────────────────────────────────────────────────
    "chest_pain": [
        r"\bchest (pain|hurts?|ache|discomfort|tightness|pressure)\b",
        r"\bpain in (my )?chest\b", r"\bheart (pain|hurts?|ache)\b",
        r"\bheartpain\b", r"\bmy chest (hurts?|is aching|is painful|is killing me)\b",
        r"\bpain (around|in|near) (my )?heart\b", r"\bcardiac pain\b",
    ],
    "chest_tightness": [
        r"\bchest (is )?tight\b", r"\btight (chest|feeling in chest)\b",
        r"\bchest (tightness|constriction|pressure|squeezing)\b",
        r"\bfeeling of pressure in (my )?chest\b", r"\bsomething (sitting|pressing|squeezing) on (my )?chest\b",
    ],
    "left_arm_pain": [
        r"\bleft arm (pain|hurts?|ache|numb|tingling|heavy)\b",
        r"\bpain in (my )?left arm\b", r"\bleft arm feels? (heavy|numb|tingly|strange)\b",
        r"\bpain radiating (to|down|into) (my )?left arm\b",
        r"\bleft shoulder (pain|ache|hurts?)\b", r"\barm (pain|ache) on (the )?left\b",
    ],
    "jaw_pain": [
        r"\bjaw (pain|hurts?|ache|tight|sore)\b", r"\bpain in (my )?jaw\b",
        r"\bpain spreading (to|into) (my )?jaw\b", r"\bjaw (is )?aching\b",
        r"\bache in (my )?jaw\b", r"\bpain.{0,30}jaw\b", r"\bjaw.{0,20}pain\b",
    ],
    "cold_sweat": [
        r"\bcold sweat(s)?\b", r"\bsweat(ing)? (cold|profusely|a lot)\b",
        r"\bwoke up? (in a )?cold sweat\b", r"\bclammy\b",
        r"\bskin (is )?clammy\b", r"\bcold and sweaty\b", r"\bbreaking (out in|into) (a )?cold sweat\b",
    ],
    "rapid_heartbeat": [
        r"\bheart (is )?racing\b", r"\brapid (heart|heartbeat|pulse)\b",
        r"\bpalpitation(s)?\b", r"\bheart beat(ing)? fast\b", r"\btachycardia\b",
        r"\bheart pounding\b", r"\bheart (is )?fluttering\b",
        r"\bpulse (is )?fast\b", r"\bfast heartbeat\b",
    ],
    "irregular_heartbeat": [
        r"\birregular (heartbeat|pulse|heart)\b", r"\bheart (is )?irregular\b",
        r"\barrhythmia\b", r"\bheart (is )?skipping\b", r"\bskipped? (a )?beat\b",
        r"\bfluttering heart\b", r"\bheart (is )?fluttering\b",
        r"\bheart (beating|beating) (irregularly|unevenly|strangely)\b",
        r"\bmissed? heartbeat\b",
    ],

    # ── Respiratory ─────────────────────────────────────────────────────────
    "cough": [
        r"\bcough(ing)?\b", r"\bhacking\b", r"\bdry cough\b", r"\bpersistent cough\b",
        r"\bchronic cough\b", r"\bkeep coughing\b", r"\bcant stop coughing\b",
    ],
    "breathing_difficulty": [
        r"\bdifficult(y| time) breathing\b", r"\bbreath(ing)? (problem|difficult|trouble|hard|short)\b",
        r"\bhard to breathe\b", r"\bcan.?t\b breathe\b", r"\bshortness of breath\b",
        r"\bcan.?t\b get (enough )?air\b", r"\btroubl.{0,10}breath\b",
        r"\bbreathless(ness)?\b", r"\bgasping\b", r"\bout of breath\b",
        r"\bwinded\b", r"\bcan.?t\b catch (my )?breath\b", r"\bsob\b",
        r"\bshort of breath\b", r"\bair (hunger|hunger)\b",
    ],
    "wheezing": [
        r"\bwheez(ing)?\b", r"\bwhistling (when|while) breath\b",
        r"\bnoisy breathing\b", r"\bbreath(ing)? (sounds? like|makes?) (a )?whistle\b",
    ],

    # ── GI ──────────────────────────────────────────────────────────────────
    "nausea": [
        r"\bnausea\b", r"\bnauseous\b", r"\bqueasy\b",
        r"\bsick to (my )?stomach\b", r"\bfeel(ing)? sick\b",
        r"\bupset stomach\b", r"\bstomach (is )?upset\b",
        r"\bwant to vomit\b", r"\bfeel(ing)? like (I might |going to )?vomit\b",
        r"\bgag(ging)?\b", r"\bbilious\b",
    ],
    "vomiting": [
        r"\bvomit(ing)?\b", r"\bthrow(ing)? up\b", r"\bthrew up\b",
        r"\bpuking?\b", r"\bpuked?\b", r"\bregurgitat\b",
        r"\bkeep(ing)? throwing up\b", r"\bcan.?t\b keep (food|anything) down\b",
    ],
    "diarrhea": [
        r"\bdiarrhea\b", r"\bdiarrhoea\b", r"\bloose stool(s)?\b",
        r"\bwatery stool(s)?\b", r"\bfrequent (stool|bowel|going to bathroom)\b",
        r"\bloose motions?\b", r"\brunning stomach\b", r"\brunny (stool|poop)\b",
        r"\bpoo(ing)? (a lot|too much|liquid|water)\b", r"\bstomach (is )?running\b",
    ],
    "constipation": [
        r"\bconstipat\b", r"\bcannot (pass|have) (stool|bowel)\b",
        r"\bno bowel movement\b", r"\bcan.?t\b (poop|go to bathroom|pass stool)\b",
        r"\bhard stool(s)?\b", r"\bstraining (to poop|to pass stool)\b",
        r"\bhaven.?t\b (pooped|had a bowel movement)\b",
    ],
    "abdominal_pain": [
        r"\babdominal (pain|ache|hurt|cramp|discomfort)\b",
        r"\bstomach (pain|ache|hurt|cramp|is killing me)\b",
        r"\bbelly (pain|ache|hurt|cramp)\b",
        r"\bpain in (my )?(stomach|abdomen|belly|gut|tummy)\b",
        r"\btummy (pain|ache|hurt|cramp)\b", r"\bgut (pain|ache|cramp)\b",
        r"\bmy stomach (hurts?|aches?|is killing me)\b",
    ],
    "bloating": [
        r"\bbloat(ed|ing)\b", r"\babdominal distension\b",
        r"\bstomach (is )?bloated\b", r"\bgassy\b", r"\bgas (pain|pressure)\b",
        r"\bstomach (feels? |is )?(full|swollen|puffed up|extended)\b",
        r"\bfull feeling in (my )?stomach\b",
    ],
    "belching": [
        r"\bbelch(ing)?\b", r"\bburp(ing)?\b", r"\bburps? (a lot|constantly|frequently)\b",
    ],
    "heartburn": [
        r"\bheartburn\b", r"\bacid reflux\b", r"\bindigestion\b",
        r"\bregurgitat\b", r"\bacid (coming|going) up\b",
        r"\bburning (in|sensation in) (my )?(chest|throat|stomach)\b",
        r"\bacid taste in (my )?mouth\b", r"\bburning (in my throat|feeling in chest)\b",
    ],
    "sour_taste": [
        r"\bsour taste\b", r"\bacid taste\b", r"\bbitter taste\b",
        r"\btaste of acid\b", r"\bacidy (taste|mouth)\b",
    ],
    "appetite_loss": [
        r"\bno appetite\b", r"\bloss of appetite\b", r"\bnot hungry\b",
        r"\bcan.?t\b eat\b", r"\bappetite (loss|is gone|reduced)\b",
        r"\bdon.?t\b feel like eating\b", r"\bnot (eating|wanting to eat)\b",
        r"\bno interest in (food|eating)\b",
    ],
    "blood_in_stool": [
        r"\bblood in (my )?(stool|poop|feces|faeces)\b",
        r"\bbloody (stool|poop|bowel movement)\b",
        r"\bblood (from|in|on) (my )?(backside|rear|anus|bottom)\b",
        r"\bblack (stool|poop|tarry stool)\b", r"\bmelena\b", r"\brectal bleeding\b",
        r"\bblood (when|after) (I poop|going to toilet|bowel movement)\b",
    ],

    # ── Urinary ─────────────────────────────────────────────────────────────
    "frequent_urination": [
        r"\bfrequent(ly)? urinat", r"\burinat(e|ing) (a lot|often|frequently)\b",
        r"\bpee (a lot|often|frequently|constantly)\b", r"\burge to urinat\b",
        r"\burinary frequency\b",
        r"\bgo (to the bathroom|to pee) (all the time|a lot|constantly)\b",
        r"\bhave to (pee|urinate) (all the time|a lot|constantly|frequently)\b",
        r"\bfeel(ing)? like (I need|having) to (urinate|pee)\b",
        r"\burination\b", r"\bpeeing (a lot|too much|frequently|often)\b",
        r"\bkeep (peeing|going to toilet|going to bathroom)\b",
    ],
    "burning_urination": [
        r"\bburn.{0,20}urinat", r"\burinat.{0,20}burn",
        r"\bpainful urinat", r"\bdysuria\b",
        r"\bit (hurts?|burns?) (when|to) (pee|urinate)\b",
        r"\bpain (when|while) (peeing|urinating)\b",
        r"\bpee (is |is )?burning\b", r"\bstinging (when|while) (urinating|peeing)\b",
        r"\bpainful pee\b", r"\bhurts? to pee\b",
        r"\bburn(ing)? (when|while) (I )?(pee|urinat)\b",
        r"\bpee.{0,20}burn(ing)?\b", r"\bburn(ing)?.{0,20}pee\b",
    ],
    "blood_in_urine": [
        r"\bblood in (my )?urine\b", r"\burinat(e|ing) blood\b",
        r"\bbloody urine\b", r"\bhematuria\b",
        r"\bpee.{0,10}blood\b", r"\bblood.{0,10}pee\b",
        r"\bred (urine|pee)\b", r"\bblood when (I pee|I urinate|peeing|urinating)\b",
    ],
    "cloudy_urine": [
        r"\bcloudy urine\b", r"\burine (is )?cloudy\b",
        r"\bstrong.{0,15}(smell|odor).{0,15}urine\b",
        r"\burine.{0,15}(smell|odor|strong|murky|milky)\b",
        r"\bmurky (urine|pee)\b", r"\bpee (is |is )?cloudy\b",
    ],
    "dark_urine": [
        r"\bdark urine\b", r"\burine (is )?(dark|brown|tea.?colored|cola.?colored)\b",
        r"\bbrown (urine|pee)\b", r"\bpee (is )?(dark|brown)\b",
    ],
    "pelvic_pain": [
        r"\bpelvic (pain|ache|pressure|discomfort)\b",
        r"\blower abdomen.{0,10}(pain|hurt|ache|pressure)\b",
        r"\bpain in (my )?(lower abdomen|pelvis|lower belly)\b",
        r"\bcramps? in (my )?(lower abdomen|pelvic area|pelvis)\b",
    ],

    # ── Skin ────────────────────────────────────────────────────────────────
    "itching": [
        r"\bitch(ing|y)?\b", r"\bscratching\b",
        r"\bskin (is )?itch(y|ing)\b", r"\bwant to scratch\b",
        r"\bcan.?t\b stop scratching\b", r"\bpruritis\b",
    ],
    "rash": [
        r"\brash(es)?\b", r"\bred spots?\b",
        r"\bspot(s)? on (my )?(skin|arm|leg|face|body|chest|back)\b",
        r"\bskin (rash|eruption|breakout|outbreak)\b", r"\bhives\b",
        r"\bwelt(s)?\b", r"\bred (patches|marks) on (my )?(skin|body)\b",
    ],
    "swelling": [
        r"\bswelling\b", r"\bswollen\b", r"\bpuff(y|iness)\b",
        r"\bpuffy (ankles?|feet|face|legs?|eyes?)\b",
        r"\bswollen (ankles?|feet|legs?|face|hands?|eyes?)\b",
        r"\bedema\b", r"\bfluid retention\b", r"\bmy (ankles?|feet|legs?) (are |is )?(swollen|puffy)\b",
    ],
    "blisters": [
        r"\bblister(s)?\b", r"\bvesicle(s)?\b",
        r"\bfluid.?filled (sore|bump|blister)\b", r"\bwater blister(s)?\b",
    ],
    "pus": [
        r"\bpus\b", r"\bfluid coming out (of )?(sore|wound|blister)\b",
        r"\bseeping?\b", r"\bweeping (sore|wound)\b",
        r"\byellow (discharge|fluid) (from|coming from) (wound|sore|skin)\b",
        r"\binfected (wound|sore|cut)\b",
    ],
    "nodules": [
        r"\bnodule(s)?\b", r"\blump(s)?\b", r"\bbump(s)?\b",
        r"\bgrowth on (my )?(skin|body)\b",
    ],
    "skin_scaling": [
        r"\bscal(y|ing|es)\b", r"\bsilvery (scale|patch|plaque)\b",
        r"\bflak(y|ing)\b", r"\bskin (is )?flak(y|ing)\b",
        r"\bpeeling skin\b", r"\bskin (is )?peeling\b",
    ],
    "skin_discoloration": [
        r"\bdifferent colou?r\b", r"\bdiscolou?r(ed|ation)\b",
        r"\bpatches? of skin.{0,20}(colou?r|color)\b", r"\bpigment\b",
        r"\bwhite patches?\b", r"\bdark patches?\b", r"\bskin (turned|become) (darker|lighter|different)\b",
    ],
    "dry_skin": [
        r"\bdry (skin|patches?)\b", r"\bskin (is |is )?dry\b",
        r"\bcracked skin\b", r"\bskin (is )?cracking\b",
        r"\bdehydrated skin\b", r"\bskin (feels? |is )?rough\b",
        r"\bneed to moisturiz\b",
    ],
    "pallor": [
        r"\bpale (skin|face|complexion|lips)\b", r"\bskin (is |is )?(pale|white|pallid|washed out)\b",
        r"\blooking pale\b", r"\bpallor\b", r"\bface (is |is )?white\b",
        r"\bno color in (my )?face\b", r"\bappear(ing)? pale\b", r"\bghostly (pale|white)\b",
        r"\bpale\b", r"\bpallid\b", r"\bashen\b",
    ],
    "nail_changes": [
        r"\bnail.{0,20}(pit|deform|thicken|dent|change|discolor)\b",
        r"\bnail(s)? (are )?(pitted|deformed|thickened|dented|yellow|brittle)\b",
        r"\b(dents?|pits?|ridges?).{0,15}nail\b", r"\bnail.{0,15}(dents?|pits?|ridges?)\b",
    ],
    "hair_loss": [
        r"\bhair (loss|falling|fall(ing)? out|thinning|dropping|shedding)\b",
        r"\bhair (is |are )?(falling|thinning|dropping|shedding|coming out)\b",
        r"\blosing (my )?hair\b", r"\bbald(ing|ness)?\b", r"\balopecia\b",
        r"\bhair (is )?thinning\b", r"\bclumps? of hair (falling|coming) out\b",
    ],
    "yellow_skin": [
        r"\byellow (skin|eyes?|urine|body)\b", r"\bskin (is |is )?yellow\b",
        r"\bjaundic\b", r"\byellow(ish)? complexion\b",
        r"\bskin (turned|become) yellow\b", r"\bwhites of (my )?eyes? (are |are )?yellow\b",
    ],

    # ── Musculoskeletal ─────────────────────────────────────────────────────
    "pain": [
        r"\bpain\b", r"\bache\b", r"\baching\b", r"\bhurts?\b", r"\bdiscomfort\b",
        r"\bsore(ness)?\b",
    ],
    "muscle_aches": [
        r"\bmuscle (ache|pain|sore|hurt)\b", r"\bmyalgia\b",
        r"\bmuscles (hurt|ache|are sore|are painful)\b",
        r"\bsore muscles?\b", r"\bachy muscles?\b",
    ],
    "muscle_stiffness": [
        r"\bstiff (muscles?|body|neck|back)\b", r"\bmuscle(s)? (are )?stiff\b",
        r"\brigid muscles?\b", r"\bstiffness (in|of) (my )?muscles?\b",
        r"\bcan.?t\b (move|bend|stretch) (my )?(muscle|body|back)\b",
        r"\bstiff (in the morning|when I wake up)\b",
    ],
    "body_aches": [
        r"\bbody (ache|hurt|pain)\b", r"\bache(s)? all over\b",
        r"\bachy\b", r"\beverything (hurts?|aches?)\b",
        r"\bmy whole body (hurts?|aches?|is sore)\b", r"\bgeneral (pain|ache)\b",
        r"\ball over pain\b",
    ],
    "joint_pain": [
        r"\bjoint (pain|ache|hurt|sore|swollen)\b",
        r"\bjoints (hurt|ache|are sore|are painful|are swollen)\b",
        r"\bpain in (my )?(joint|knee|hip|elbow|shoulder|wrist|ankle|knuckle)\b",
        r"\bknee (pain|hurts?|ache|sore)\b", r"\bhip (pain|hurts?|ache)\b",
        r"\bshoulder (pain|hurts?|ache)\b", r"\bwrist (pain|hurts?|ache)\b",
        r"\bankle (pain|hurts?|ache)\b",
    ],
    "joint_stiffness": [
        r"\bjoint(s)? (stiffness|stiff|are stiff)\b",
        r"\bstiff (joint|knee|hip|elbow|shoulder|ankle|wrist)\b",
        r"\bjoints.{0,20}stiff\b", r"\bstiff.{0,20}joint\b",
        r"\bhard to move\b", r"\bdifficult(y| time) moving\b",
        r"\bcan.?t\b move (my )?(joint|knee|hip|shoulder|elbow)\b",
        r"\bdifficult.{0,20}walk\b", r"\bmorning stiffness\b",
    ],
    "back_pain": [
        r"\bback (pain|hurts?|ache|sore)\b", r"\bbackache\b",
        r"\bpain in (my )?back\b", r"\bback is (sore|painful|killing me)\b",
        r"\blower back (pain|ache|hurts?)\b", r"\bupper back (pain|ache|hurts?)\b",
        r"\bspine (pain|ache|hurts?)\b",
    ],
    "neck_pain": [
        r"\bneck (pain|hurts?|ache|sore|stiff)\b", r"\bstiff neck\b",
        r"\bpain in (my )?neck\b", r"\bcervical pain\b",
        r"\bneck (is )?stiff\b", r"\bmy neck (hurts?|is sore|is killing me)\b",
    ],
    "flank_pain": [
        r"\bflank (pain|ache|hurts?)\b",
        r"\bpain in (my )?(side|flank|right side|left side|lower back)\b",
        r"\bside (pain|ache|hurts?)\b", r"\bkidney pain\b",
        r"\bpain (in|on) (my )?(left|right) side\b",
        r"\bpain (going|radiating) (from|to) (my )?(back|side)\b",
    ],
    "leg_cramps": [
        r"\bleg cramp(s)?\b", r"\bcramp(s)? in (my )?(leg|calf|calves)\b",
        r"\bcalf cramp(s)?\b", r"\bcharley horse\b",
        r"\bleg (is |is )?(cramping|seized up)\b",
    ],
    "visible_veins": [
        r"\bvisible vein(s)?\b", r"\bvaricose\b",
        r"\bvein(s)? on (my )?(leg|arm|calf)\b", r"\benlarged vein(s)?\b",
        r"\bbulging vein(s)?\b",
    ],

    # ── Metabolic / Endocrine ───────────────────────────────────────────────
    "excessive_thirst": [
        r"\bthirst(y)?\b", r"\bexcessive thirst\b", r"\bvery thirsty\b",
        r"\bdrink(ing)? (a lot|too much|constantly)\b", r"\bpolydipsia\b",
        r"\bcan.?t\b quench (my )?thirst\b", r"\balways (thirsty|want(ing)? water)\b",
    ],
    "excessive_hunger": [
        r"\bexcessive hunger\b", r"\balways hungry\b", r"\bincreased appetite\b",
        r"\bpolyphagia\b", r"\bexcessive appetite\b",
        r"\bcan.?t\b stop eating\b", r"\bstarving (all the time|constantly)\b",
        r"\bhungry (all the time|constantly|even after eating)\b",
    ],
    "slow_healing": [
        r"\bwounds? (not|slow to|won.t) heal\b", r"\bslow (healing|recovery)\b",
        r"\bcuts?.{0,10}(not|slow) heal\b", r"\bskin not healing\b",
        r"\bsores? (not|won.t) heal\b", r"\btakes? (forever|long time) to heal\b",
    ],
    "heat_intolerance": [
        r"\bsensitiv(e|ity) to heat\b", r"\bcan.?t\b stand (the )?heat\b",
        r"\bheat (intolerance|sensitive|bothers?|intolerant)\b",
        r"\bfeel(ing)? too hot (all the time|always)\b", r"\boverheating\b",
        r"\balways (feel(ing)? )?too hot\b",
    ],
    "cold_intolerance": [
        r"\bsensitiv(e|ity) to cold\b", r"\bcan.?t\b stand (the )?cold\b",
        r"\bcold (intolerance|sensitive|bothers?|intolerant)\b",
        r"\bfeel(ing)? (too )?cold (all the time|always)\b",
        r"\balways (feel(ing)? )?cold\b", r"\bcan.?t\b get warm\b",
    ],
    "cold_extremities": [
        r"\bcold (hands?|feet|toes?|fingers?)\b",
        r"\bhands? (are |feel )?(cold|freezing|icy|numb)\b",
        r"\bfeet (are |feel )?(cold|freezing|icy|numb)\b",
        r"\bpoor circulation\b", r"\bfingers?.{0,10}(cold|numb)\b",
        r"\bhands? and feet (are |are )?cold\b", r"\bmy (hands?|feet|fingers?) go numb (from cold|in the cold)\b",
    ],

    # ── Infection markers ───────────────────────────────────────────────────
    "bleeding": [
        r"\bbleed(ing)?\b",
        r"\bblood in (my )?(urine|stool|vomit|sputum|phlegm)\b",
        r"\bhemorrhag\b", r"\bcough(ing)? up blood\b",
        r"\bblood (from|out of) (my )?(mouth|nose)\b",
        r"\bnosebleed\b",
    ],

    # ── Sensory ─────────────────────────────────────────────────────────────
    "phonophobia": [
        r"\bsensitiv(e|ity) to sound\b", r"\bnoise sensitive\b",
        r"\bbothered by (noise|sound)\b", r"\bsound(s)? (hurt|bother|are too loud)\b",
        r"\bcan.?t\b stand (loud |any )?noise\b",
    ],
}

# Pre-compiled patterns for fast matching
_COMPILED = {
    sym: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for sym, pats in SYMPTOM_PATTERNS.items()
}


def match_symptoms(text, valid_symptoms=None):
    """
    Match natural-language symptom text to canonical KG symptom names.
    If valid_symptoms is provided (a set), only return names that exist in the KG.
    """
    matched = set()
    for sym, patterns in _COMPILED.items():
        if valid_symptoms is not None and sym not in valid_symptoms:
            continue
        for pat in patterns:
            if pat.search(text):
                matched.add(sym)
                break
    return list(matched)
