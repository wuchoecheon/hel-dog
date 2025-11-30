

def calc_sleep_score(sleep_logs):
    s = sum([75 + x.label * 5 for x in sleep_logs])
    l = max(len(sleep_logs), 1)
    return (s//l)