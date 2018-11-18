import itertools
import time
import warnings
from collections import defaultdict
from typing import List, Optional, Sequence, Dict, Tuple, Any
from xml.etree import ElementTree

import speedtest

__author__ = "Bojan Potočnik"


class SpeedtestServer:
    def __init__(self, xml_server_node: ElementTree.Element):
        self.attrib = xml_server_node.attrib

    def __str__(self) -> str:
        return "{}({})".format(type(self).__name__, self.attrib)

    def __repr__(self) -> str:
        return "<{} at 0x{:08x}".format(str(self), id(self))

    @property
    def id(self) -> int:
        return int(self.attrib["id"])

    @property
    def name(self) -> str:
        return self.attrib["name"]

    @property
    def url(self) -> str:
        return self.attrib["url"]

    @property
    def host(self) -> str:
        return self.attrib["host"]

    @property
    def country(self) -> str:
        return self.attrib["country"]

    @property
    def country_code(self) -> str:
        return self.attrib["cc"]

    @property
    def sponsor(self) -> str:
        return self.attrib["sponsor"]

    @property
    def latitude(self) -> float:
        return float(self.attrib["lat"])

    @property
    def longitude(self) -> float:
        return float(self.attrib["lon"])

    @property
    def lat_lon(self) -> Tuple[float, float]:
        """:return: The server location tuple (latitude, longitude)"""
        return self.latitude, self.longitude

    def distance_to(self, lat_lon: Tuple[float, float]) -> float:
        """
        Calculate the difference to the other coordinate (server) and also save it to the parameters.

        :param lat_lon: Location of the other server (latitude, longitude).

        :return: Distance in kilometers.
        """
        self.attrib["d"] = speedtest.distance(self.lat_lon, lat_lon)
        return self.attrib["d"]

    def speedtest_server(self, st: speedtest.Speedtest) -> Tuple[float, Dict[str, Any]]:
        """
        Get server representation compatible with speedtest.Speedtest().servers dictionary.

        :param st: Speedtest instance to retrieve the config.

        :return: Key and value, ready to be appended to the speedtest.Speedtest().servers dictionary.
        """
        d = self.distance_to(st.lat_lon)
        return d, self.attrib


def get_servers(*,
                countries: Optional[Sequence[str]] = None,
                country_codes: Optional[Sequence[str]] = None,
                names: Optional[Sequence[str]] = None,
                hosts: Optional[Sequence[str]] = None,
                host_contains: Optional[Sequence[str]] = None,
                sponsors: Optional[Sequence[str]] = None,
                from_cache: bool = False, cache_servers: bool = True) -> Optional[List[SpeedtestServer]]:
    import os

    cache_file = "cache/servers.xml"

    if from_cache and os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            servers_xml = f.read().decode()
    else:
        # If servers are cached, there is no need for `requests` package.
        import requests

        resp: requests.Response = requests.get(r"http://www.speedtest.net/speedtest-servers.php")
        if resp.ok and resp.text:
            servers_xml = resp.text
            if cache_servers:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    f.write(servers_xml.encode())
        else:
            print(resp.status_code, resp.reason)
            servers_xml = None
    if servers_xml:
        try:
            xml_tree: ElementTree.Element = ElementTree.fromstringlist(servers_xml)
            # Parse all servers
            servers: List[SpeedtestServer] = []
            for srv in xml_tree[0]:  # type: ElementTree.Element
                servers.append(SpeedtestServer(srv))
        except ElementTree.ParseError as e:
            warnings.warn(e)
            servers = None
    else:
        servers = None

    if servers:
        # Use generators for filtering, create list only once at the end.
        if countries:
            servers = filter(lambda server: (server.country in countries), servers)
        if names:
            servers = filter(lambda server: (server.name in names), servers)
        if country_codes:
            servers = filter(lambda server: (server.country_code in country_codes), servers)
        if hosts:
            servers = filter(lambda server: (server.host in hosts), servers)
        if host_contains:
            servers = filter(lambda server: any(host in server.host for host in host_contains), servers)
        if sponsors:
            servers = filter(lambda server: (server.sponsor in sponsors), servers)
        # If any filter was provided, servers will be filter type now.
        if not isinstance(servers, list):
            servers = list(servers)

    return servers


def speedtest_servers(st: Optional[speedtest.Speedtest] = None, servers: Optional[Sequence[SpeedtestServer]] = None) \
        -> List[speedtest.SpeedtestResults]:
    if not st:
        st = speedtest.Speedtest()
    if servers:
        # Use custom servers instead of `get_servers()`
        st.servers = defaultdict(list)
        for server in servers:
            k, v = server.speedtest_server(st)
            st.servers[k].append(v)
    else:
        st.get_servers()
        st.get_best_server()
        st.servers = st.best  # FIXME
    # Test to all servers.
    all_servers = list(itertools.chain.from_iterable(st.servers.values()))
    results = []
    for server in all_servers:
        print("Testing server {}".format(server))
        start_time = time.perf_counter()
        st.best = server
        # Test speeds
        st.download()
        st.upload()
        st.results.server = server
        results.append(st.results)
        print("Done in {:.3f} s: {}".format(time.perf_counter() - start_time, st.results.dict()))

    return results


def main():
    custom_servers = get_servers(countries=("Slovenia",), names=("Ljubljana",), from_cache=True)
    print("Testing to servers:")
    for server in custom_servers:
        print(" - {}".format(server))

    out_file = "results.csv"
    with open(out_file, "a"):
        pass

    while True:
        results = speedtest_servers(servers=custom_servers)

        print("Saving results to {}".format(out_file))
        with open(out_file, "a") as f:
            f.writelines(str(result.dict()) + "\n" for result in results)

        time.sleep(5 * 60)


def test():
    s = speedtest.Speedtest()
    s.get_servers()
    best_server = s.get_best_server()
    print("best", best_server)
    s.download()
    s.upload()
    # s.results.share()

    results_dict = s.results.dict()
    print(results_dict)
    print(s.results.csv())


if __name__ == "__main__":
    # test()
    main()
