"""M8: rule-based natural-language parser and M3-M7 chart dispatcher."""
import re
import pandas as pd
from .shared import PARAMETER_COLUMNS, apply_filters
from .m3_mobility import prepare_mobility_analysis, render_mobility_analysis
from .m4_geospatial import prepare_geospatial_analysis, render_geospatial_analysis
from .m5_distributions import prepare_distribution_analysis, render_distribution_analysis
from .m6_relationships import prepare_relationship_analysis, render_relationship_analysis
from .m7_multivariate import prepare_multivariate_analysis, render_multivariate_analysis

# M8 rule-based vocabulary. Aliases normalize to canonical notebook fields and chart types.
M8_WORD_ALIASES = {
    "rng": "range", "yr": "year", "yrs": "year", "uni": "university",
    "univ": "university", "inst": "institution", "institutes": "institution",
    "institutions": "institution", "cat": "category", "categ": "category",
    "engg": "engineering", "eng": "engineering", "mgmt": "management",
    "pharma": "pharmacy", "med": "medical", "arch": "architecture",
    "agri": "agriculture", "corr": "correlation", "rel": "relationship",
    "hist": "histogram", "scat": "scatter", "hmap": "heatmap",
    "quart": "quartile", "spid": "radar", "geo": "geospatial", "choro": "choropleth", "lin": "line",
    "stats": "statistics",
}

M8_PARAMETER_ALIASES = {
    "tlr": ["tlr", "teaching learning resources", "teaching"],
    "rpc": ["rpc", "research and professional practice", "research"],
    "go": ["go", "graduation outcomes", "graduation outcome"],
    "oi": ["oi", "outreach and inclusivity", "outreach", "inclusivity"],
    "pr": ["pr", "perception"],
    "score": ["overall scores", "overall score", "scores", "score"],
    "rank": ["ranks", "rank", "ranking"],
}

M8_CATEGORY_ALIASES = {
    "Overall": ["overall"], "Engineering": ["engineering"],
    "Management": ["management"], "College": ["college"],
    "University": ["university"], "Pharmacy": ["pharmacy"],
    "Architecture": ["architecture"], "Law": ["law"],
    "Medical": ["medical"], "Dental": ["dental"],
    "Research": ["research category"], "Agriculture": ["agriculture"],
    "State Public University": ["state public university", "spu"],
}

M8_CHART_ROUTE = {
    "trend": "mobility", "transition": "mobility", "transition_percentage": "mobility",
    "movement": "mobility", "improvement": "mobility", "decline": "mobility",
    "largest_movement": "mobility", "top10": "mobility",
    "map": "geospatial", "state_bar": "geospatial", "state_trend": "geospatial",
    "state_compare": "geospatial", "new_entries": "geospatial",
    "histogram": "distribution", "box": "distribution", "violin": "distribution",
    "ecdf": "distribution", "quartile": "distribution", "stats_table": "distribution",
    "top10_vs_rest": "distribution", "by_year": "distribution", "median_iqr": "distribution",
    "by_state": "distribution", "parameter_variability": "distribution",
    "scatter": "relationship", "matrix": "relationship", "score_correlations": "relationship", "scatter_matrix": "relationship",
    "correlation_trend": "relationship", "parameter_trend": "relationship",
    "radar": "multivariate", "parallel": "multivariate", "gap": "multivariate",
    "research_teaching": "multivariate", "teaching_perception": "multivariate",
    "parameter_distribution": "multivariate", "outlier": "multivariate",
    "score_change": "multivariate", "score_gain_parameter_decline": "multivariate",
    "research_score_relative_outlier": "multivariate", "parameter_rank": "multivariate",
    "volatility": "multivariate", "gap_outlier": "multivariate",
}

def parse_question(question, df=None):
    """Parse common phrasing, abbreviations, filters, and chart cues without an external API."""
    import re
    if not isinstance(question, str) or not question.strip():
        return {"status": "unsupported", "message": "Enter a question to analyze."}

    text = question.lower().strip().translate(str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    }))
    text = re.sub(r"\buniversities\b", "university", text)
    text = re.sub(r"\binstitutes\b|\binstitutions\b", "institution", text)
    for alias, canonical in M8_WORD_ALIASES.items():
        text = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, text)
    text = re.sub(r"\s+", " ", text)

    params = {}
    filters = {}
    range_match = re.search(
        r"\b(?:range|between|from)\s+(20\d{2})\s*(?:to|through|thru|and|[-–])\s*(20\d{2})\b", text
    )
    if not range_match:
        range_match = re.search(r"\b(20\d{2})\s*[-–]\s*(20\d{2}|\d{2})\b", text)
    years = []
    if range_match:
        first, last = int(range_match.group(1)), int(range_match.group(2))
        if last < 100:
            last = (first // 100) * 100 + last
            if last < first:
                last += 100
        if first > last:
            return {"status": "invalid_input", "message": "The year range must start before it ends. Enter years from 2017 to 2025."}
        if last - first <= 30:
            years = list(range(first, last + 1))
            params["start_year"], params["end_year"] = first, last
    if not years:
        years = sorted({int(y) for y in re.findall(r"\b20\d{2}\b", text)})
    if df is not None and "year" in df.columns:
        available_years = pd.to_numeric(df["year"], errors="coerce").dropna()
        min_year = int(available_years.min()) if not available_years.empty else 2017
        max_year = int(available_years.max()) if not available_years.empty else 2025
    else:
        min_year, max_year = 2017, 2025
    invalid_years = [year for year in years if year < min_year or year > max_year]
    if invalid_years:
        return {
            "status": "invalid_input",
            "message": f"Year(s) {', '.join(map(str, invalid_years))} are outside the available range. Enter year(s) between {min_year} and {max_year}.",
        }
    if years:
        filters["year"] = years

    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90, "hundred": 100,
    }
    number_pattern = r"(\d+|" + "|".join(sorted(number_words, key=len, reverse=True)) + r")"
    top_matches = list(re.finditer(r"\btop\s*" + number_pattern + r"\b", text))
    for top_match in top_matches:
        token = top_match.group(1)
        count = int(token) if token.isdigit() else number_words[token]
        following = text[top_match.end():top_match.end() + 30]
        if re.match(r"\s*(?:ranked\s+)?states?\b", following):
            params["top_states"] = count
        elif re.match(r"\s*(?:ranked\s+)?(?:institutions?|universities|university|unis?)\b", following):
            params["top_n"] = count
        else:
            params.setdefault("top_n", count)

    if re.search(r"\bspearman\b", text):
        params["method"] = "spearman"
    elif re.search(r"\bpearson\b", text):
        params["method"] = "pearson"
    params["degree"] = int(re.search(r"\bdegree\s*(\d+)\b", text).group(1)) if re.search(r"\bdegree\s*(\d+)\b", text) else 2

    # Parameter names in the question become M6 scatter axes when two are stated.
    variable_hits = []
    for column, aliases in M8_PARAMETER_ALIASES.items():
        for alias in aliases:
            if column in ("go", "oi", "pr") and alias in ("go", "oi", "pr"):
                # Avoid treating ordinary words like "go" as NIRF acronyms; full names still work.
                if not re.search(r"\b" + re.escape(alias.upper()) + r"\b", question):
                    continue
            match = re.search(r"\b" + re.escape(alias) + r"\b", text)
            if match:
                variable_hits.append((match.start(), column))
                break
    variable_hits = sorted({(position, column) for position, column in variable_hits})
    variables = []
    for _, column in variable_hits:
        if column not in variables:
            variables.append(column)
    if len(variables) >= 2:
        params["x_var"], params["y_var"] = variables[0], variables[1]
        params["columns"] = list(dict.fromkeys(variables + ["score"]))

    # Detect canonical institution names and common national abbreviations from the loaded data.
    institution_matches = []
    if df is not None and "institution" in df.columns:
        names = [str(name) for name in df["institution"].dropna().unique()]
        alias_map = {}
        prefixes = [
            ("indian institute of technology ", "iit "),
            ("indian institute of management ", "iim "),
            ("national institute of technology ", "nit "),
            ("all india institute of medical sciences ", "aiims "),
        ]
        for name in names:
            alias_map.setdefault(name.lower(), set()).add(name)
            lower_name = name.lower()
            for full_prefix, short_prefix in prefixes:
                if lower_name.startswith(full_prefix):
                    alias_map.setdefault(short_prefix + lower_name[len(full_prefix):], set()).add(name)
            if lower_name == "indian institute of science":
                alias_map.setdefault("iisc", set()).add(name)
        for alias, canonical_names in alias_map.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", text):
                for canonical in canonical_names:
                    institution_matches.append((len(alias), canonical))
    institution_names = sorted({name for _, name in institution_matches})
    if institution_names:
        longest = max(length for length, _ in institution_matches)
        filters["institution"] = sorted({name for length, name in institution_matches if length == longest})

    # Resolve categories using dataset values plus common short forms.
    category_values = list(df["category"].dropna().unique()) if df is not None and "category" in df.columns else list(M8_CATEGORY_ALIASES)
    category_hits = []
    category_cue = bool(re.search(r"\bcategory\b|\btype\b", text))
    for category in category_values:
        if (
            category == "Overall" and not category_cue
            and re.search(r"\boverall\s+(?:score|rank|ranking|performance)\b", text)
        ):
            # Here "overall" describes the measure, not the NIRF category.
            continue
        aliases = M8_CATEGORY_ALIASES.get(category, [str(category).lower()])
        if any(re.search(r"\b" + re.escape(alias) + r"\b", text) for alias in aliases):
            category_hits.append(category)
    # Do not mistake a category word inside a matched institution name for a category filter.
    if len(category_hits) == 1 and (category_cue or not institution_names):
        filters["category"] = category_hits
    elif len(category_hits) > 1 and category_cue:
        return {"status": "needs_clarification", "message": f"More than one category matched: {category_hits}. Please specify one."}
    elif category_cue and not category_hits:
        valid_categories = ", ".join(sorted(map(str, category_values)))
        return {
            "status": "invalid_input",
            "message": f"That category is not available. Enter one of: {valid_categories}.",
        }

    # Extract full state names, state codes, or two-state comparisons.
    state_values = list(df["state"].dropna().unique()) if df is not None and "state" in df.columns else []
    state_codes = {}
    if df is not None and {"state", "state_code"}.issubset(df.columns):
        state_codes = dict(df[["state_code", "state"]].dropna().drop_duplicates().itertuples(index=False, name=None))
    state_hits = []
    compare_words = bool(re.search(r"\b(?:vs|versus|compare|comparison|between)\b", text))
    for state in state_values:
        if re.search(r"\b" + re.escape(str(state).lower()) + r"\b", text):
            state_hits.append(state)
    if state_codes:
        for code, state in state_codes.items():
            if re.search(r"\b" + re.escape(str(code).lower()) + r"\b", text):
                state_hits.append(state)
    state_hits = list(dict.fromkeys(state_hits))
    state_context = bool(re.search(r"\b(?:state|states|in|from|across|regional|region)\b", text))
    if len(state_hits) >= 2 and compare_words:
        filters["state"] = state_hits
        params["states"] = state_hits
    elif len(state_hits) == 1 and state_context:
        filters["state"] = state_hits
    if institution_names and state_hits and df is not None and "institution_id" in df.columns:
        candidates = df[df["institution"].isin(institution_names) & df["state"].isin(state_hits)]
        candidate_ids = candidates["institution_id"].dropna().unique().tolist()
        if len(candidate_ids) == 1:
            filters["institution_id"] = candidate_ids
        elif len(candidate_ids) > 1:
            matched_states = sorted(candidates["state"].dropna().astype(str).unique())
            return {"status": "needs_clarification", "message": f"That institution name matches multiple records in {', '.join(matched_states)}. Please specify the state."}

    # Choose module/chart from explicit concepts and chart cues; never guess on mixed chart requests.
    has = lambda pattern: bool(re.search(pattern, text))
    chart_type = None
    if has(r"\bpercent(?:age)?\b.*\bsame\s+(?:rank\s+)?band\b"):
        chart_type = "transition_percentage"
    elif has(r"\b(?:51\s*[-–]\s*100|51\s+to\s+100)\b.*\btop\s*50\b"):
        chart_type = "transition"
    elif has(r"\b(?:band\s+)?transition|\benter(?:ed|ing)?\s+(?:or\s+)?exit|\bup\s+down\s+same\b"):
        chart_type = "transition_percentage" if has(r"percent(?:age)?") and has(r"same\s+(?:rank\s+)?band") else "transition"
    elif has(r"\bmoved?\s+up\b") and has(r"\bmoved?\s+down\b") and has(r"\b(?:retained|stayed|same)\b"):
        chart_type = "movement"
    elif has(r"\b(?:largest|greatest|biggest)\b") and has(r"\b(?:upward|downward|up|down)\b") and has(r"\brank\s+movements?\b"):
        chart_type = "largest_movement"
    elif has(r"\b(?:rank|ranking)\b") and has(r"\b(?:largest|greatest|biggest)\b") and has(r"\b(?:declin\w*|drop\w*|fall\w*)\b"):
        chart_type = "decline"
    elif has(r"\b(?:rank|ranking)\b") and has(r"\b(?:greatest|most|improv\w*|gain\w*)\b") and has(r"\b(?:over|across|between|from|year|period)\b"):
        chart_type = "improvement"
    elif has(r"\bnew\s+entr(?:y|ies)|\bnewly\s+appearing\b") and (state_hits or has(r"state|region")):
        chart_type = "new_entries"
    elif has(r"\b(?:correlation|relationship)\s+(?:matrix|heatmap)|\bheatmap\b|\bhmap\b"):
        chart_type = "matrix"
    elif has(r"\bscatter\s+matrix|\bpairs?\s+plot\b"):
        chart_type = "scatter_matrix"
    elif has(r"\bstate\s+(?:compare|comparison)|\bstates?\s+(?:vs|versus)|\b(?:vs|versus|compare)\b") and (state_hits or has(r"state|karnataka|tamil nadu")):
        chart_type = "state_compare"
    elif has(r"\b(?:choropleth(?:\s+map)?|(?:bubble\s+)?map|geospatial|geo\s+map)\b"):
        chart_type = "map"
    elif has(r"\bgeograph\w*\b") and has(r"\b(?:distribution|chang\w*|shift|top.rank\w*)\b"):
        chart_type = "map"
    elif has(r"\bstates?\b") and has(r"\b(?:improv\w*|increas\w*|chang\w*)\b") and has(r"\b(?:representat\w*|over time|trend)\b"):
        chart_type = "state_trend"
    elif has(r"\bstates?\s+(?:trends?|over time)|\bregional\s+shift"):
        chart_type = "state_trend"
    elif has(r"\b(?:average|avg|mean)\b.*\bscore\b") and has(r"\b(?:over time|by year|trend|chang\w*)\b"):
        chart_type = "parameter_trend"
        params["columns"] = ["score"]
    elif has(r"\bdistribution\b.*\b(?:selected|given|specific)\s+year\b") and has(r"\bscore|rank\b"):
        chart_type = "histogram"
    elif has(r"\b(?:distribution|spread|dispersed|dispersion)\b.*\byears?\b|\bby year\b") and has(r"\b(?:spread|distribution|score|dispersed|dispersion)\b"):
        chart_type = "by_year"
    elif has(r"\bbox(?:\s*plot|plot)?\b.*\bstates?\b|\bscore\s+distribution\b.*\bstates?\b|\bby state\b.*\b(?:score|rank)\b"):
        chart_type = "by_state"
    elif has(r"\b(?:spread|distribution)\b.*\b(?:across|between)\b.*\bcategor(?:y|ies)\b|\bscore spread\b.*\bcategor(?:y|ies)\b"):
        chart_type = "box"
    elif has(r"\bstates?\b") and has(r"\b(?:highest|largest|greatest|most)\b") and has(r"\b(?:number|count|institutions?)\b"):
        chart_type = "state_bar"
    elif has(r"\bstate\s+bar|\bstate\s+count|\bby\s+state|\bstate\s+distribution"):
        chart_type = "state_bar"
    elif has(r"\b(?:hist|histogram)\b"):
        chart_type = "histogram"
    elif has(r"\bviolin\b"):
        chart_type = "violin"
    elif has(r"\bbox(?:\s*plot|plot)?\b"):
        chart_type = "box"
    elif has(r"\becdf\b|\bcumulative distribution\b"):
        chart_type = "ecdf"
    elif has(r"\bquart(?:ile)?s?\b"):
        chart_type = "quartile"
    elif has(r"\btop\s*(?:10|ten)\b.*\b(?:remaining|rest|other institutions?)\b"):
        chart_type = "top10_vs_rest"
    elif has(r"\bmedian\b.*\b(?:iqr|interquartile)\b|\biqr\b"):
        chart_type = "median_iqr"
    elif has(r"\b(?:summary statistics|stats table|statistics table|statistics)\b|\b(?:average|avg|mean|median|minimum|maximum|min|max)\b.*\b(?:score|rank)\b"):
        chart_type = "stats_table"
    elif has(r"\btop\s*(?:10|ten)\s+(?:vs|versus|and)\s+(?:the\s+)?rest\b"):
        chart_type = "top10_vs_rest"
    elif has(r"\bparameter\s+variability|\bmost\s+variable\s+parameter|\bparameter\b.*\b(?:greatest|highest|largest)\s+variability"):
        chart_type = "parameter_variability"
    elif has(r"\bcorrelation\b") and has(r"\b(?:strongest|highest|largest)\b") and has(r"\bparameter\b") and has(r"\boverall\s+score\b"):
        chart_type = "score_correlations"
    elif has(r"\bparameter\s+gap(?:s)?\b.*\bcategor(?:y|ies)\b|\bparameter\s+distribution\b"):
        chart_type = "parameter_distribution"
    elif has(r"\b(?:correlation|corr|relationship)\b") and has(r"\b(?:over time|by year|trend|changed)\b"):
        chart_type = "correlation_trend"
    elif has(r"\bscore\b") and has(r"\b(?:trend|over time|by year|chang\w*)\b"):
        # A general score trend means the yearly mean score for the filtered
        # population (for example, all institutions in a named state).
        chart_type = "parameter_trend"
        params["columns"] = ["score"]
    elif has(r"\b(?:average|avg|mean)\b.*\bscore\b.*\b(?:over time|by year|trend)\b"):
        chart_type = "parameter_trend"
        params["columns"] = ["score"]
    elif has(r"\baverage\s+parameter\b|\bparameter\s+scores?\s+over time"):
        chart_type = "parameter_trend"
    elif has(r"\b(?:correlation|corr)\b") and len(variables) < 2:
        chart_type = "matrix"
    elif has(r"\b(?:correlat|associat|relationship|relate)\w*\b") and len(variables) >= 2:
        chart_type = "scatter"
    elif has(r"\bscatter\b|\bscat\b|\bplot\b.*\bvs\b|\bvs\b.*\bscore\b"):
        chart_type = "scatter"
    elif has(r"\bradar|\bspider\b"):
        chart_type = "radar"
    elif has(r"\bparallel\b"):
        chart_type = "parallel"
    elif has(r"\bvolatility|\bvolatile\b"):
        chart_type = "volatility"
    elif has(r"\b(?:score|scores)\b") and has(r"\b(?:year.to.year|over time|unusually large|unusual change)\b") and has(r"\b(?:chang\w*|increas\w*|decreas\w*)\b"):
        chart_type = "score_change"
    elif has(r"\bimprov\w*\b") and has(r"\boverall\s+scores?\b") and has(r"\bdeclin\w*\b") and has(r"\bparameter"):
        chart_type = "score_gain_parameter_decline"
    elif has(r"\b(?:strong|high)\b") and has(r"\bparameter\b") and has(r"\b(?:poor|low|weak)\b") and has(r"\brank|ranking\b"):
        chart_type = "parameter_rank"
    elif has(r"\bresearch\b") and has(r"\b(?:overall|score)\b") and has(r"\b(?:unusual|high|relative)\b"):
        chart_type = "research_score_relative_outlier"
    elif has(r"\bresearch\b") and has(r"\bteaching\b") and has(r"\b(?:better|higher|substantially|substantially better|gap)\b"):
        chart_type = "research_teaching"
    elif has(r"\bteaching\b") and has(r"\bperception|\bpr\b") and has(r"\b(?:high|higher|low|lower|better|worse)\b"):
        chart_type = "teaching_perception"
    elif has(r"\boutlier|\bunusual\b"):
        chart_type = "gap_outlier" if has(r"gap|teaching|research") else "outlier"
    elif has(r"\bscore\s+change|\byear.to.year\s+score"):
        chart_type = "score_change"
    elif has(r"\bresearch\s+(?:vs|versus|compared with|and)\s+teaching|\bteaching\s+(?:vs|versus)\s+research"):
        chart_type = "research_teaching"
    elif has(r"\bteaching\s+(?:vs|versus|and)\s+perception|\bperception\s+(?:vs|versus)\s+teaching"):
        chart_type = "teaching_perception"
    elif has(r"\bparameter\s+gap|\bstrongest\s+(?:and|vs)\s+weakest\s+parameter"):
        chart_type = "gap"
    elif has(r"\btop\s*(?:10|ten)\b") and has(r"\b(?:consistent|across|multiple years|institutions)\b"):
        chart_type = "top10"
    elif has(r"\b(?:rank|ranking)\b") and has(r"\b(?:chang(?:e|ed|ing)|evolv|from)\b"):
        chart_type = "trend"
    elif has(r"\b(?:rank|ranking)\b") and has(r"\b(?:movement|mobility|improv\w*|declin\w*|trend|over time)\b"):
        chart_type = "trend" if has(r"trend|over time") else "movement"
    elif has(r"\btop\s*(?:5|five)\b.*\b(?:compare|parameter|institution)\b|\bcompare\b.*\btop\s*(?:5|five)\b"):
        chart_type = "radar"
    elif has(r"\bresearch\b") and has(r"\bteaching\b") and has(r"\b(?:high|higher|low|lower|gap|better|worse)\b"):
        chart_type = "research_teaching"
    elif has(r"\bteaching\b") and has(r"\bperception\b") and has(r"\b(?:high|higher|low|lower|gap|better|worse)\b"):
        chart_type = "teaching_perception"
    elif has(r"\bline\s+chart\b") and has(r"\bstate|\bregional\b"):
        chart_type = "state_trend"
    elif has(r"\bline\s+chart\b") and has(r"\bcorrelation|\brelationship\b"):
        chart_type = "correlation_trend"
    elif has(r"\bline\s+chart\b") and has(r"\bparameter\b"):
        chart_type = "parameter_trend"
    elif has(r"\bline\s+chart\b") and has(r"\brank\b"):
        chart_type = "trend"
    elif has(r"\bdistribution|\bspread\b"):
        chart_type = "histogram"
    elif has(r"\bstate|\bregional|\bgeograph"):
        chart_type = "map"

    if chart_type is None:
        if has(r"\bbar\s+chart\b"):
            return {"status": "needs_clarification", "message": "Which result should the bar chart compare: state counts, parameter variability, or another summary?"}
        if has(r"\bline\s+chart\b|\bline\b"):
            return {"status": "needs_clarification", "message": "What should the line chart show: rank, state counts, parameter scores, or correlation over time?"}
        if has(r"\b(?:nirf|rank|ranking|score|parameter|institution|institutions|school|schools|college|colleges|university|universities|state|research|teaching|correlation|distribution|rpc|tlr|perception|pharmacy|engineering|management|overall|architecture|dental|medical|agriculture|law)\b"):
            return {
                "status": "needs_clarification",
                "message": "I recognized NIRF data terms, but need the task more specifically. Are you asking for a trend, comparison, distribution, correlation, state summary, or outlier analysis?",
            }
        return {"status": "unsupported", "message": "I could not identify an M3-M7 analysis. Try naming a chart or question, such as 'hist RPC', 'state choropleth', or 'RPC vs score scatter'."}

    intent = M8_CHART_ROUTE[chart_type]
    if chart_type == "state_trend":
        if has(r"\b(?:improv\w*|increas\w*|grow\w*|growth)\b"):
            params["state_order"] = "increase"
        elif has(r"\b(?:declin\w*|decreas\w*|drop\w*|fall\w*)\b"):
            params["state_order"] = "decrease"
    asks_for_selected_institution = bool(
        re.search(r"\bselected\s+(?:institution|university)\b", text)
    )
    if (
        chart_type == "trend"
        and asks_for_selected_institution
        and not institution_names
        and not filters.get("institution")
    ):
        return {
            "status": "needs_input",
            "message": "Please enter the institution name whose rank trend you want to see.",
            "intent": intent,
            "chart_type": chart_type,
            "filters": filters,
            "params": params,
        }
    if chart_type == "state_compare" and not params.get("states"):
        params["states"] = ["Karnataka", "Tamil Nadu"]
    if chart_type in ("scatter", "trend") and intent == "relationship":
        if len(variables) >= 2:
            params.setdefault("x_var", variables[0])
            params.setdefault("y_var", variables[1])
        elif chart_type == "scatter":
            return {"status": "needs_clarification", "message": "Name two numeric variables for the scatter chart, such as RPC and score."}
        else:
            params.setdefault("x_var", "rpc")
            params.setdefault("y_var", "score")

    value_column = next((v for _, v in variable_hits if v in ("score", "rank", *PARAMETER_COLUMNS)), "score")
    if intent == "distribution":
        params["value_column"] = value_column
    if (
        chart_type in {"histogram", "stats_table"}
        and has(r"\b(?:selected|given|specific)\s+year\b")
        and not filters.get("year")
    ):
        params["requires_single_year"] = True

    return {
        "status": "ready", "intent": intent, "chart_type": chart_type,
        "filters": filters, "params": params, "normalized_question": text,
        "question_focus": (
            "which" if re.search(r"\bwhich\b", text)
            else "when" if re.search(r"\bwhen\b", text)
            else None
        ),
        "question_entity": (
            "state" if re.search(r"\bstates?\b", text)
            else "parameter" if re.search(r"\bparameters?\b", text)
            else "institution" if re.search(r"\binstitution|university|college\b", text)
            else None
        ),
    }


def dispatch_question(parsed, df, filters=None):
    """Validate an M8 parse, call its M3-M7 preparation/render pair, and return the figure."""
    if parsed.get("status") != "ready":
        return {"status": parsed.get("status", "unsupported"), "message": parsed.get("message", "Question was not parsed.")}

    intent, chart_type = parsed["intent"], parsed["chart_type"]
    params = dict(parsed.get("params", {}))
    merged_filters = dict(filters or {})
    merged_filters.update(parsed.get("filters", {}))  # explicit question filters take precedence
    requested_years = merged_filters.get("year")
    if requested_years is not None:
        requested_years = requested_years if isinstance(requested_years, (list, tuple, set)) else [requested_years]
        scope_filters = {key: value for key, value in merged_filters.items() if key != "year"}
        scope = apply_filters(df, scope_filters)
        present_years = set(pd.to_numeric(scope["year"], errors="coerce").dropna().astype(int).unique())
        missing_years = [int(year) for year in requested_years if int(year) not in present_years]
        if missing_years:
            is_requested_range = {"start_year", "end_year"}.issubset(params)
            if is_requested_range:
                available_in_range = sorted(year for year in requested_years if int(year) in present_years)
                if chart_type in {"transition", "transition_percentage", "improvement", "decline"}:
                    endpoints = [int(params["start_year"]), int(params["end_year"])]
                    absent_endpoints = [year for year in endpoints if year not in present_years]
                    if absent_endpoints:
                        return {
                            "status": "unavailable",
                            "message": f"No records were found for the requested transition endpoint(s): {', '.join(map(str, absent_endpoints))}. No chart was generated.",
                            "alternatives": ["Choose endpoint years with records for the selected institution/category.", "Use a rank trend to show the years that are available."],
                        }
                elif available_in_range:
                    # Ranges may include years in which a selected institution or
                    # category has no record. Plot observed years without inventing
                    # zero values; trend renderers still require multiple points.
                    merged_filters["year"] = available_in_range
                else:
                    return {
                        "status": "unavailable",
                        "message": f"No records were found between {params['start_year']} and {params['end_year']} with the selected filters. No chart was generated.",
                        "alternatives": ["Choose another category, state, or institution that has records in this period.", "Use a year range with available data."],
                    }
            else:
                available = sorted(present_years)
                return {
                    "status": "unavailable",
                    "message": f"No records were found for {', '.join(map(str, missing_years))} with the selected filters. No chart was generated.",
                    "alternatives": [
                        f"Use available years in this selection: {available[:3]}{'...' if len(available) > 3 else ''}.",
                        "Choose another category, state, or institution that has records for those years.",
                        "Choose a different year range or ask for a summary of the available period.",
                    ],
                }

    # A few member charts require explicit scope to avoid invalid comparisons.
    categories = merged_filters.get("category")
    categories = categories if isinstance(categories, (list, tuple, set)) else ([categories] if categories else [])
    years = merged_filters.get("year")
    years = years if isinstance(years, (list, tuple, set)) else ([years] if years is not None else [])
    if params.get("requires_single_year") and len(years) != 1:
        return {"status": "needs_input", "message": "Select one year in the dashboard filters, or include a year such as 2024 in your question."}
    top_states = params.get("top_states")
    if top_states is not None and chart_type in ("state_trend", "state_bar"):
        try:
            top_states = int(top_states)
        except (TypeError, ValueError):
            top_states = 0
        state_scope = apply_filters(df, merged_filters)
        available_states = state_scope["state"].dropna().nunique() if "state" in state_scope else 0
        if top_states < 1 or top_states > available_states:
            return {
                "status": "invalid_input",
                "message": f"Invalid number of states: {top_states}. Enter a number between 1 and {available_states} for the selected filters.",
            }
    if chart_type == "radar" and (len(categories) != 1 or len(years) != 1):
        return {"status": "needs_input", "message": "The radar chart needs one NIRF category and one year. Please specify both."}
    if chart_type in {"improvement", "decline", "largest_movement", "parameter_rank"} and len(categories) != 1:
        return {"status": "needs_input", "message": "This rank comparison is category-specific. Select one NIRF category, such as Overall or College."}
    if chart_type == "volatility" and len(categories) != 1:
        return {"status": "needs_input", "message": "The volatility chart needs one NIRF category. Please specify a category."}
    if chart_type == "state_compare" and len(categories) != 1:
        return {"status": "needs_input", "message": "State rank comparison needs one NIRF category. Please specify a category."}
    if intent == "relationship" and chart_type in ("trend", "correlation_trend", "parameter_trend") and len(years) == 1:
        return {"status": "needs_input", "message": "A relationship trend needs at least two years. Give a year range or clear the year filter."}

    routes = {
        "mobility": (prepare_mobility_analysis, render_mobility_analysis),
        "geospatial": (prepare_geospatial_analysis, render_geospatial_analysis),
        "distribution": (prepare_distribution_analysis, render_distribution_analysis),
        "relationship": (prepare_relationship_analysis, render_relationship_analysis),
        "multivariate": (prepare_multivariate_analysis, render_multivariate_analysis),
    }
    if intent not in routes or chart_type not in M8_CHART_ROUTE or M8_CHART_ROUTE[chart_type] != intent:
        return {"status": "unsupported", "message": "That chart route is not available in the current M3-M7 notebook."}

    prepare, render = routes[intent]
    answer_year_pair = None
    if intent == "mobility":
        # M3 needs prior years to calculate a movement; calculate history first,
        # then keep only the years the user requested for display.
        mobility_filters = {key: value for key, value in merged_filters.items() if key != "year"}
        result = prepare(df, mobility_filters)
        endpoint_charts = {"improvement", "decline", "largest_movement"}
        if chart_type in endpoint_charts and {"start_year", "end_year"}.issubset(params):
            result = result[
                result["year"].between(int(params["start_year"]), int(params["end_year"]))
            ].copy()
        elif requested_years is not None and chart_type not in endpoint_charts and chart_type not in {"transition", "transition_percentage"}:
            result = result[result["year"].isin(requested_years)].copy()
        if chart_type in {"trend", "year_trend_bar"}:
            ranked_rows = result[pd.to_numeric(result["rank"], errors="coerce").ge(1)]
            identity = (["category"] if "category" in ranked_rows else []) + ["institution_id" if "institution_id" in ranked_rows else "institution"]
            enough_history = not ranked_rows.empty and ranked_rows.groupby(identity)["year"].nunique().ge(2).any()
            if not enough_history:
                years_available = sorted(map(int, pd.to_numeric(ranked_rows["year"], errors="coerce").dropna().astype(int).unique()))
                return {
                    "status": "unavailable",
                    "message": f"A rank trend needs ranked records in at least two years. Available year(s) for this selection: {years_available or 'none'}. No chart was generated.",
                    "alternatives": ["Choose an institution/category with records in multiple years.", "Ask for a single-year rank or score summary instead."],
                }
        render_params = {key: params[key] for key in ("start_year", "end_year", "top_n") if key in params}
        if chart_type in {"transition", "transition_percentage"} and not {"start_year", "end_year"}.issubset(render_params):
            if len(years) >= 2:
                render_params.update(start_year=int(min(years)), end_year=int(max(years)))
            elif re.search(r"\bconsecutive\s+years?\b|\bone\s+year\s+to\s+the\s+next\b", parsed.get("normalized_question", "")):
                year_scope = apply_filters(df, {key: value for key, value in merged_filters.items() if key != "year"})
                available = sorted(pd.to_numeric(year_scope["year"], errors="coerce").dropna().astype(int).unique())
                pairs = [(first, second) for first, second in zip(available, available[1:]) if second == first + 1]
                if pairs:
                    render_params.update(start_year=int(pairs[-1][0]), end_year=int(pairs[-1][1]))
                else:
                    return {"status": "needs_input", "message": "No consecutive ranked years are available for this selection. Specify a year range."}
            else:
                return {"status": "needs_input", "message": "A rank-band transition needs a start and end year. Specify a range."}
        if chart_type == "movement" and not result["movement"].isin(["up", "down", "same"]).any():
            return {"status": "unavailable", "message": "No adjacent-year rank comparison is available for the selected year(s). No chart was generated.", "alternatives": ["Include a later year with ranked records.", "Choose a later year range.", "Use a rank trend chart for the selected year(s)."]}
        if chart_type in {"transition", "transition_percentage"}:
            answer_year_pair = (int(render_params["start_year"]), int(render_params["end_year"]))
        figure = render(result, chart_type=chart_type, **render_params)
    elif intent == "geospatial":
        result = prepare(df, merged_filters, top_n=params.get("top_n", 100))
        if chart_type == "state_trend" and result["state_summary"]["year"].nunique() < 2:
            return {"status": "needs_input", "message": "A state trend needs at least two years. Give a year range or clear the Year filter."}
        render_params = {"chart_type": chart_type}
        render_params["top_states"] = params.get(
            "top_states", 10 if chart_type == "state_bar" else 8
        )
        if chart_type == "state_trend":
            render_params["state_order"] = params.get("state_order", "representation")
        if chart_type == "state_compare":
            render_params["states"] = params.get("states", ["Karnataka", "Tamil Nadu"])
        figure = render(result, **render_params)
    elif intent == "distribution":
        result = prepare(df, merged_filters, value_column=params.get("value_column", "score"))
        if chart_type == "median_iqr" and result["data"]["year"].nunique() < 2:
            return {"status": "unavailable", "message": "A median and IQR trend needs data in at least two years. No chart was generated.", "alternatives": ["Choose a wider year range or a category with records in multiple years."]}
        figure = render(result, chart_type=chart_type)
    elif intent == "relationship":
        prep_params = {"method": params.get("method", "pearson")}
        if "columns" in params:
            prep_params["columns"] = params["columns"]
        result = prepare(df, merged_filters, **prep_params)
        if chart_type in {"trend", "correlation_trend", "parameter_trend"} and result["data"]["year"].nunique() < 2:
            return {"status": "unavailable", "message": "A relationship trend needs valid data in at least two years. No chart was generated.", "alternatives": ["Choose a wider year range or clear the Year filter.", "Choose another category or variable pair with data across multiple years."]}
        render_params = {"chart_type": chart_type}
        for key in ("x_var", "y_var", "degree"):
            if key in params:
                render_params[key] = params[key]
        if "trendline" in params:
            render_params["trendline"] = params["trendline"]
        figure = render(result, **render_params)
    else:
        result = prepare(df, merged_filters)
        figure = render(result, chart_type=chart_type)

    # Build a short answer from the plotted traces, so question words such as
    # "which" and "when" return concrete names/years alongside the figure.
    focus = parsed.get("question_focus")
    answer = None
    answer_table_override = None
    answer_rows = []
    if focus:
        for trace in figure.data:
            trace_x = getattr(trace, "x", None)
            trace_y = getattr(trace, "y", None)
            if trace_x is None and trace_y is None:
                # Plotly table traces expose cells rather than x/y coordinates.
                continue
            xs = list(trace_x) if trace_x is not None else []
            ys = list(trace_y) if trace_y is not None else []
            series_name = str(trace.name) if trace.name else ""
            for index, y_value in enumerate(ys):
                x_value = xs[index] if index < len(xs) else None
                answer_rows.append({"series": series_name, "year_or_group": x_value, "value": y_value})

        if focus == "which":
            names = []
            if intent == "distribution" and chart_type == "median_iqr":
                value_column = result["value_column"]
                measure_label = "overall score" if value_column == "score" else f"{value_column.upper()} score"
                yearly = result["data"].groupby("year")[value_column].agg(
                    q1=lambda values: values.quantile(.25),
                    median="median",
                    q3=lambda values: values.quantile(.75),
                ).dropna(subset=["median"])
                if not yearly.empty:
                    yearly["iqr"] = yearly["q3"] - yearly["q1"]
                    best = yearly[yearly["median"].eq(yearly["median"].max())]
                    answer = "; ".join(
                        f"{int(year)} had the highest median {measure_label} "
                        f"({row['median']:.2f}); its IQR was {row['iqr']:.2f} "
                        f"(Q1 {row['q1']:.2f} to Q3 {row['q3']:.2f})"
                        for year, row in best.iterrows()
                    ) + "."
                    answer_table_override = best.reset_index().rename(columns={"q1": "Q1", "median": "Median", "q3": "Q3", "iqr": "IQR"})
            if intent == "mobility" and chart_type in {"transition", "transition_percentage"} and answer_year_pair:
                start_year, end_year = answer_year_pair
                ranked = result.copy()
                ranked["rank"] = pd.to_numeric(ranked["rank"], errors="coerce")
                valid = ranked["rank"].ge(1)
                if "rank_status" in ranked:
                    valid &= ranked["rank_status"].ne("Rank withdrawn")
                identity = (["category"] if "category" in ranked.columns else []) + ["institution_id" if "institution_id" in ranked.columns else "institution"]
                display_identity = [c for c in ["institution", "state"] if c in ranked and c not in identity]
                start_top = ranked.loc[
                    valid & ranked["year"].eq(start_year) & ranked["rank"].le(10),
                    identity + display_identity + ["rank"],
                ].rename(columns={"rank": "rank_at_start"})
                end_top = ranked.loc[
                    valid & ranked["year"].eq(end_year) & ranked["rank"].le(10),
                    identity + display_identity + ["rank"],
                ].rename(columns={"rank": "rank_at_end"})
                start_keys = set(map(tuple, start_top[identity].itertuples(index=False, name=None)))
                end_keys = set(map(tuple, end_top[identity].itertuples(index=False, name=None)))
                entered_keys = end_keys - start_keys
                exited_keys = start_keys - end_keys
                entered = end_top[
                    end_top[identity].apply(lambda row: tuple(row) in entered_keys, axis=1)
                ].copy()
                exited = start_top[
                    start_top[identity].apply(lambda row: tuple(row) in exited_keys, axis=1)
                ].copy()
                entered["transition"] = f"Entered Top 10 in {end_year}"
                exited["transition"] = f"Exited Top 10 after {start_year}"
                transition_rows = pd.concat([entered, exited], ignore_index=True, sort=False)
                if "state" in transition_rows and transition_rows["institution"].duplicated(keep=False).any():
                    transition_rows["institution"] = transition_rows["institution"].astype(str) + " — " + transition_rows["state"].fillna("Unknown state").astype(str)
                if "category" in transition_rows and transition_rows["category"].nunique() > 1:
                    transition_rows["institution"] = (
                        transition_rows["institution"].astype(str)
                        + " (" + transition_rows["category"].astype(str) + ")"
                    )
                transition_rows = transition_rows[
                    ["transition", "institution", *(["category"] if "category" in transition_rows else []),
                     *(["rank_at_start"] if "rank_at_start" in transition_rows else []),
                     *(["rank_at_end"] if "rank_at_end" in transition_rows else [])]
                ]
                answer_table_override = transition_rows
                entered_names = transition_rows.loc[
                    transition_rows["transition"].str.startswith("Entered"), "institution"
                ].astype(str).tolist()
                exited_names = transition_rows.loc[
                    transition_rows["transition"].str.startswith("Exited"), "institution"
                ].astype(str).tolist()
                answer = (
                    f"Top-10 changes from {start_year} to {end_year}. "
                    f"Entered: {', '.join(entered_names) if entered_names else 'none'}. "
                    f"Exited: {', '.join(exited_names) if exited_names else 'none'}."
                )
            if intent == "geospatial" and chart_type in {"state_bar", "state_trend"} and parsed.get("question_entity") == "state":
                summary = result["state_summary"].copy()
                if chart_type == "state_trend":
                    years_scope = sorted(summary["year"].dropna().astype(int).unique())
                    states_scope = sorted(summary.loc[summary["institutions"].gt(0), "state"].dropna().astype(str).unique())
                    grid = pd.MultiIndex.from_product([years_scope, states_scope], names=["year", "state"]).to_frame(index=False)
                    summary = grid.merge(summary[["year", "state", "institutions"]], on=["year", "state"], how="left").fillna({"institutions": 0}).sort_values("year")
                    state_totals = summary.groupby("state").agg(first=("institutions", "first"), last=("institutions", "last"), start_year=("year", "first"), end_year=("year", "last")).reset_index()
                    state_totals["change"] = state_totals["last"] - state_totals["first"]
                    increasing = params.get("state_order") != "decrease"
                    state_totals = state_totals[state_totals["change"].gt(0) if increasing else state_totals["change"].lt(0)]
                    state_totals = state_totals.sort_values("change", ascending=not increasing).head(int(params.get("top_states", 8)))
                    direction = "increase" if increasing else "decrease"
                    answer = (f"States with the largest {direction} in representation: " + "; ".join(f"{r.state} ({int(r.change):+d}; {int(r.first)} to {int(r.last)} institutions, {int(r.start_year)}–{int(r.end_year)})" for r in state_totals.itertuples(index=False))) if not state_totals.empty else f"No states increased their representation over {years_scope[0]}–{years_scope[-1]}."
                    answer_table_override = state_totals
                else:
                    state_totals = summary.groupby("state", as_index=False)["institutions"].sum().rename(columns={"institutions": "top_n_institution_years"}).sort_values("top_n_institution_years", ascending=False).head(int(params.get("top_states", 10)))
                    per_year = summary[summary["state"].isin(state_totals["state"])].merge(state_totals, on="state", how="left").sort_values(["top_n_institution_years", "year"], ascending=[False, True])
                    n_label = result.get("top_n") or "all ranked"
                    answer = f"States with the highest Top-{n_label} representation across the selected years: " + "; ".join(f"{row.state} ({int(row.top_n_institution_years)} institution-year appearances)" for row in state_totals.itertuples(index=False))
                    answer_table_override = per_year.rename(columns={"institutions": f"institutions_in_top_{n_label}", "top_n_institution_years": "selected_year_total"})
            if intent == "geospatial" and chart_type == "new_entries" and parsed.get("question_entity") == "state":
                entries = result["entry_summary"].copy()
                state_totals = entries.groupby("state", as_index=False)["new_entries"].sum().sort_values("new_entries", ascending=False).head(int(params.get("top_states", 10)))
                answer = "States with the most newly appearing institutions: " + "; ".join(f"{r.state} ({int(r.new_entries)})" for r in state_totals.itertuples(index=False))
                answer_table_override = state_totals
            if intent == "relationship" and chart_type == "score_correlations" and parsed.get("question_entity") == "parameter":
                correlations = result["correlation"]["score"].drop(labels=["score"], errors="ignore").dropna().sort_values(key=lambda values: values.abs(), ascending=False)
                answer_table_override = correlations.rename_axis("Parameter").rename("Correlation with overall score").reset_index()
                if not correlations.empty:
                    answer = f"{correlations.index[0].upper()} has the strongest absolute correlation with overall score (r = {correlations.iloc[0]:.2f})."
            # For institution-level lines, each Plotly trace is an institution.
            if intent == "mobility" and chart_type == "trend":
                names.extend(str(trace.name) for trace in figure.data if trace.name)
            if intent == "mobility" and chart_type == "top10":
                for trace in figure.data:
                    if trace.x is not None:
                        names.extend(str(value) for value in trace.x if value is not None)
            # Horizontal ranking charts put institution names on the category axis.
            if intent == "mobility" and chart_type in {"improvement", "decline", "largest_movement"}:
                for trace in figure.data:
                    if trace.y is not None:
                        names.extend(str(value) for value in trace.y if value is not None)
            if intent == "multivariate" and chart_type in {"score_gain_parameter_decline", "volatility"}:
                for trace in figure.data:
                    values = trace.y if chart_type == "score_gain_parameter_decline" else trace.x
                    if values is not None:
                        names.extend(str(value) for value in values if value is not None)
            # M4 entry counts can also name the newly appearing institutions.
            if intent == "geospatial" and chart_type == "new_entries" and parsed.get("question_entity") != "state":
                new_rows = result["all_data"]
                if "is_new_entry" in new_rows:
                    mask = new_rows["is_new_entry"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
                    names.extend(new_rows.loc[mask, "institution"].dropna().astype(str).tolist())
            # For M7 outlier charts, report only institutions flagged by the
            # same condition used to colour the chart.
            if intent == "multivariate" and isinstance(result, dict) and "data" in result:
                flag = {
                    "outlier": "rpc_outlier",
                    "research_score_relative_outlier": "rpc_relative_outlier",
                    "gap_outlier": "gap_outlier",
                    "score_change": "score_change_outlier",
                }.get(chart_type)
                rows = result["data"]
                if chart_type == "research_teaching":
                    flag = "research_teaching_high"
                elif chart_type == "teaching_perception":
                    flag = "teaching_perception_high"
                elif chart_type == "parameter_rank":
                    flag = "strong_parameter_poor_rank"
                elif chart_type == "gap" and "parameter_gap" in rows:
                    names.extend(rows.nlargest(20, "parameter_gap")["institution_label"].dropna().astype(str).tolist())
                if flag in rows.columns and "institution" in rows.columns:
                    flagged = rows[flag].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
                    label_column = "institution_label" if "institution_label" in rows else "institution"
                    names.extend(rows.loc[flagged, label_column].dropna().astype(str).tolist()[:20])
            # Deduplicate while preserving the chart's displayed order.
            names = list(dict.fromkeys(names))
            if answer is None and names:
                answer = "Institutions shown: " + "; ".join(names)
            elif answer is None:
                answer = "The chart does not identify individual institutions for this analysis; see its plotted series and filters."
        elif focus == "when":
            institutions = merged_filters.get("institution", [])
            institutions = institutions if isinstance(institutions, (list, tuple, set)) else [institutions]
            if not institutions and intent == "mobility" and chart_type == "trend":
                institutions = [str(trace.name) for trace in figure.data if trace.name]
            years_in_chart = sorted({
                int(float(row["year_or_group"])) for row in answer_rows
                if row["year_or_group"] is not None
                and str(row["year_or_group"]).replace(".0", "").isdigit()
                and 1900 <= int(float(row["year_or_group"])) <= 2100
            })
            if chart_type == "new_entries":
                years_in_chart = sorted({
                    int(str(trace.name)) for trace in figure.data
                    if trace.name and str(trace.name).isdigit()
                })
            subject = ", ".join(map(str, institutions)) if institutions else "the selected data"
            answer = f"For {subject}, the chart covers {', '.join(map(str, years_in_chart)) or 'the displayed period'}. See the plotted values for the relevant year(s)."

    if intent == "distribution" and chart_type == "median_iqr" and answer is None:
        value_column = result["value_column"]
        measure_label = "overall score" if value_column == "score" else f"{value_column.upper()} score"
        yearly = result["data"].groupby("year")[value_column].agg(
            q1=lambda values: values.quantile(.25),
            median="median",
            q3=lambda values: values.quantile(.75),
        ).reset_index().sort_values("year")
        yearly["iqr"] = yearly["q3"] - yearly["q1"]
        answer_table_override = yearly.rename(columns={"q1": "Q1", "median": "Median", "q3": "Q3", "iqr": "IQR"})
        if len(yearly) >= 2:
            first, last = yearly.iloc[0], yearly.iloc[-1]
            answer = (
                f"From {int(first['year'])} to {int(last['year'])}, median {measure_label} "
                f"changed from {first['median']:.2f} to {last['median']:.2f} "
                f"({last['median'] - first['median']:+.2f}); IQR changed from "
                f"{first['iqr']:.2f} to {last['iqr']:.2f} ({last['iqr'] - first['iqr']:+.2f})."
            )
    answer_table = answer_table_override if answer_table_override is not None else (pd.DataFrame(answer_rows) if answer_rows else None)
    return {
        "status": "ready", "intent": intent, "chart_type": chart_type,
        "filters": merged_filters, "message": f"Showing {chart_type.replace('_', ' ')} for the requested data.",
        "figure": figure, "result": result, "direct_answer": answer,
        "answer_table": answer_table,
    }

