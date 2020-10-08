# Adapted from https://esgf-pyclient.readthedocs.io/en/latest/notebooks/examples/download.html
# and https://github.com/ESGF/esgf-pyclient/issues/57
# and https://esgf-pyclient.readthedocs.io/en/latest/notebooks/demo/subset-cmip6.html
import os
from pyesgf.search import SearchConnection
from pyesgf.logon import LogonManager
from myproxy.client import MyProxyClient
from OpenSSL import SSL

MyProxyClient.SSL_METHOD = SSL.TLSv1_2_METHOD

openID = os.getenv("openID")
openID_password = os.getenv("openID_password")
if not openID:
    raise ValueError("openID not set")

lm = LogonManager()
lm.logon_with_openid(openID, password=openID_password)
lm.is_logged_on()


def test():
    conn = SearchConnection("https://esgf-data.dkrz.de/esg-search", distrib=True)
    ctx = conn.new_context(
        project="CMIP6",
        source_id="UKESM1-0-LL",
        experiment_id="historical",
        variable="tas",
        frequency="mon",
        variant_label="r1i1p1f2",
        data_node="esgf-data3.ceda.ac.uk",
    )
    print(ctx.hit_count)
    result = ctx.search()[0]

    files = result.file_context().search()
    for file in files:
        print(file.opendap_url)
    return True

def get_openDAP_urls(search):
    conn = SearchConnection("https://esgf-data.dkrz.de/esg-search", distrib=True)
    ctx = conn.new_context(**search)
    if not ctx.hit_count:
        raise ValueError('No data found matching search')

    result = ctx.search()[0]
    files = result.file_context().search()
    return [file.opendap_url for file in files]


if __name__ == "__main__":
    test()
