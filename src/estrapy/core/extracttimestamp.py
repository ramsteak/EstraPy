import re
import datetime
from dateutil import parser

def extract_datetime_from_file(header: str) -> tuple[datetime.datetime | None, datetime.timedelta | None]:
    # 1. CLEANUP
    # Split into lines and strip leading '#' and whitespace
    lines = [line.lstrip('#').strip() for line in header.split('\n') if line.strip()]
    full_text = " ".join(lines)

    # 2. EXTRACT TIMES (Global Search)
    # MODIFIED REGEX: Allows 'T' as a prefix separator for ISO 8601 (e.g., ...T12:00:00)
    # Matches: "12:00", "T12:00", " 12:00:00"
    time_pattern = r'(?:\b|T)(\d{1,2}:\d{2}(?::\d{2})?)Z?\b'
    found_time_strs = re.findall(time_pattern, full_text)
    
    found_times: list[datetime.time] = []
    for t_str in found_time_strs:
        try:
            found_times.append(parser.parse(t_str).time())
        except ValueError:
            continue

    # 3. EXTRACT DURATION (Global Search)
    duration_pattern = r'(?i)\b(\d+)\s*(s|sec|seconds?|m|min|minutes?|h|hours?)\b'
    found_durations = re.findall(duration_pattern, full_text)

    # 4. EXTRACT DATE (Line-by-Line Search)
    parsed_date_part = None
    current_year = datetime.datetime.now().year
    default_date = datetime.datetime(1, 1, 1)

    # Match lines with Year (1990-2029) or Month Name
    candidate_matcher = re.compile(r'(199\d|20[0-2]\d)|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', re.IGNORECASE)

    for line in lines:
        if not candidate_matcher.search(line):
            continue

        clean_line = line
        # Remove found times from the line so parsing assumes it's just a date
        for t_str in found_time_strs:
            # We must handle the 'T' case in replacement too
            # If the line contains "T03:00", simply removing "03:00" leaves "T", 
            # which dateutil usually ignores, but we clean it to be safe.
            clean_line = clean_line.replace(t_str, "")
        
        # Cleanup residual "T" if it was an ISO string (e.g., "2024-10-08T" -> "2024-10-08")
        clean_line = clean_line.replace("T", " ")
        clean_line = re.sub(duration_pattern, '', clean_line)

        try:
            dt = parser.parse(clean_line, fuzzy=True, default=default_date)
            
            # Validation: Year must exist and be reasonable
            if dt.year != 1 and (1990 <= dt.year <= current_year + 1):
                parsed_date_part = dt
                break 
        except (ValueError, TypeError, OverflowError):
            continue

    if not parsed_date_part:
        return None, None

    # 5. COMBINE
    # Case A: 2 Times (Start -> End)
    if len(found_times) >= 2:
        start_dt = datetime.datetime.combine(parsed_date_part.date(), found_times[0])
        end_dt = datetime.datetime.combine(parsed_date_part.date(), found_times[1])
        
        if end_dt < start_dt:
            end_dt += datetime.timedelta(days=1)
            
        return (start_dt, end_dt - start_dt)

    # Case B: 1 Time + Duration
    elif len(found_times) == 1 and found_durations:
        start_dt = datetime.datetime.combine(parsed_date_part.date(), found_times[0])
        val, unit = found_durations[0]
        val = int(val)
        unit = unit.lower()
        
        delta = datetime.timedelta()
        if 's' in unit: delta = datetime.timedelta(seconds=val)
        elif 'm' in unit: delta = datetime.timedelta(minutes=val)
        elif 'h' in unit: delta = datetime.timedelta(hours=val)
        
        return (start_dt, delta)

    # Case C: 1 Time
    elif len(found_times) == 1:
        return datetime.datetime.combine(parsed_date_part.date(), found_times[0]), None

    # Case D: Only Date
    else:
        return parsed_date_part, None
