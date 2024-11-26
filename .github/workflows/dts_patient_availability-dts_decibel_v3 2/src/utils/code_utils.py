from utils.db_pool import redshift_engine
from sqlalchemy import text

def get_concept_codes_for_indicator(c_indicator_names):
    if len(c_indicator_names) == 1:
        query = f"""
                select code, vocabulary from dmgmt_common_lookup.cancer_type_dx_code_lookup where cancer_type = '{c_indicator_names[0]}'
            """
    else:
        query = f"""
                    select code, vocabulary from dmgmt_common_lookup.cancer_type_dx_code_lookup where cancer_type in {tuple(c_indicator_names)}
                """
    results = redshift_engine.execute(text(query))
    results = results.fetchall()
    dic = {}
    for r in results:
        if r[1] not in dic: dic[r[1]] = [r[0]]
        else: dic[r[1]].append(r[0])
    codes = []
    for voc in dic:
        base_query = f"select concept_code from omop_vocabulary.concept where vocabulary_id = '{voc}' AND ("
        for c in dic[voc]:
            base_query += f" concept_code ilike '{c}%' OR "
        base_query = base_query[:-3]
        base_query += ')'
        results = redshift_engine.execute(text(base_query))
        results = results.fetchall()
        results = [r[0] for r in results]
        codes += results
    return codes
if __name__ == '__main__':
    print (get_concept_codes_for_indicator('Multiple Myeloma'))