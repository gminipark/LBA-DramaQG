def get_tags(ent_idx, ent_type, style="backslash"):
    ent_type = ent_type.upper()
    open_tag = f"<e{ent_idx}:{ent_type}>"
    close_tag = f"<\\e{ent_idx}:{ent_type}>" if style == "backslash" else f"</e{ent_idx}:{ent_type}>"
    return open_tag, close_tag


def tag_from_json(row, id2sample, TAG_STYLE):
    rid = row["id"]
    sample = id2sample.get(rid)

    if sample is None:
        return None 

    tokens = sample.get("token", None)
    if tokens is None:
        return None

    ss = sample["subj_start"]
    se = sample["subj_end"]
    os_ = sample["obj_start"]
    oe = sample["obj_end"]
    stype = sample["subj_type"]
    otype = sample["obj_type"]

    e1_open, e1_close = get_tags(1, stype, TAG_STYLE)
    e2_open, e2_close = get_tags(2, otype, TAG_STYLE)

    out = []
    for i, tok in enumerate(tokens):
        if i == ss:
            out.append(e1_open)
        if i == os_:
            out.append(e2_open)

        out.append(tok)

        if i == se:
            out.append(e1_close)
        if i == oe:
            out.append(e2_close)

    return " ".join(out)
