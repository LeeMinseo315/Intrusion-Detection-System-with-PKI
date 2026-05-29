import pandas as pd

# conn.log 로드
conn = pd.read_csv('conn.log', sep='\t', comment='#', header=None,
    names=['ts','uid','orig_h','orig_p','resp_h','resp_p','proto','service',
           'duration','orig_bytes','resp_bytes','conn_state','local_orig',
           'local_resp','missed_bytes','history','orig_pkts','orig_ip_bytes',
           'resp_pkts','resp_ip_bytes','tunnel_parents','ip_proto'])

# ssl.log 로드
ssl = pd.read_csv('ssl.log', sep='\t', comment='#', header=None,
    names=['ts','uid','orig_h','orig_p','resp_h','resp_p','version','cipher',
           'curve','server_name','resumed','last_alert','next_protocol',
           'established','ssl_history','cert_chain_fps','client_cert_chain_fps',
           'sni_matches_cert'])

# uid 기준으로 병합
merged = pd.merge(conn, ssl[['uid','version','cipher','curve','established']], 
                  on='uid', how='inner')

print(f"병합된 행 수: {len(merged)}")
print(merged[['uid','orig_p','resp_p','version','cipher','established']].head())

# 저장
merged.to_csv('~/merged_tls.csv', index=False)
print("저장 완료: merged_tls.csv")
