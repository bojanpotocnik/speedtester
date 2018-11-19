import datetime
import itertools
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Dict, Tuple, Any, Union
from xml.etree import ElementTree

import speedtest

__author__ = "Bojan Potočnik"


@dataclass(frozen=True)
class SpeedtestServer:
    id: int
    name: str
    url: str
    host: str
    country: str
    country_code: str
    sponsor: str
    latitude: float
    longitude: float
    xml_attrib: Dict[str, Union[str, float]] = field(repr=False)

    @classmethod
    def from_xml(cls, xml_server_node: ElementTree.Element) -> 'SpeedtestServer':
        atr = xml_server_node.attrib

        return SpeedtestServer(
            xml_attrib=atr,
            id=int(atr["id"]),
            name=atr["name"],
            url=atr["url"],
            host=atr["host"],
            country=atr["country"],
            country_code=atr["cc"],
            sponsor=atr["sponsor"],
            latitude=float(atr["lat"]),
            longitude=float(atr["lon"])
        )

    def __hash__(self) -> int:
        return self.id

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
        return speedtest.distance(self.lat_lon, lat_lon)

    def speedtest_server(self, st: speedtest.Speedtest) -> Tuple[float, Dict[str, Any]]:
        """
        Get server representation compatible with speedtest.Speedtest().servers dictionary.

        :param st: Speedtest instance to retrieve the config.

        :return: Key and value, ready to be appended to the speedtest.Speedtest().servers dictionary.
        """
        d = self.distance_to(st.lat_lon)
        self.xml_attrib["d"] = d
        return d, self.xml_attrib


def get_servers(*,
                countries: Optional[Sequence[str]] = None,
                country_codes: Optional[Sequence[str]] = None,
                names: Optional[Sequence[str]] = None,
                hosts: Optional[Sequence[str]] = None,
                host_contains: Optional[Sequence[str]] = None,
                sponsors: Optional[Sequence[str]] = None,
                from_cache: bool = False, cache_servers: bool = True,
                cache_file: str = "cache/servers.xml") -> Optional[List[SpeedtestServer]]:
    """
    Get list of available SpeedTest server.

    :param countries:     If provided, only servers from this countries will be listed.
    :param country_codes: If provided, only servers with such country codes will be listed.
    :param names:         If provided, only servers matching any of the provided names will be listed.
    :param hosts:         If provided, only servers matching any of the provided hosts will be listed.
    :param host_contains: If provided, only servers of which host contains any of the provided strings will be listed.
    :param sponsors:      If provided, only servers of which sponsor match any of the provided sponsors will be listed.

    :param from_cache:    Whether to use the cache file for server information instead
                          of fetching the list from the SpeedTest website.
    :param cache_servers: Whether to save the fetched server list to local cache file.
    :param cache_file:    Path to local cache file.

    :return: List of server or None if cache didn't exist and server list could not be fetched from the internet.
    """
    import os

    # Get list of the servers from cache or via web request
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
            print(resp.status_code, resp.reason, file=sys.stderr)
            servers_xml = None

    # Parse list of server
    if servers_xml:
        try:
            xml_tree: ElementTree.Element = ElementTree.fromstringlist(servers_xml)
            # Parse all servers
            servers: List[SpeedtestServer] = [
                SpeedtestServer.from_xml(srv) for srv in xml_tree[0]
            ]
        except ElementTree.ParseError as e:
            print(e, file=sys.stderr)
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
        print(f"{datetime.datetime.now()}: Testing server {server}")
        start_time = time.perf_counter()
        # Set this server as the best server as only this server is tested.
        st._best = server
        # Test speeds
        st.download()
        st.upload()
        st.results.server = server
        results.append(st.results)
        print(f"{datetime.datetime.now()}: Done in {time.perf_counter() - start_time:.3f} s: {st.results.dict()}")

    return results


def main():
    custom_servers = get_servers(countries=("Slovenia",), names=("Ljubljana",), from_cache=True)
    print("Testing to servers:")
    for server in custom_servers:
        print(" - {}".format(server))

    # Create the file
    out_file = "results.csv"
    with open(out_file, 'a'):
        pass

    while True:
        print(f"{datetime.datetime.now()}: Testing {len(custom_servers)} servers...")
        start_time = time.perf_counter()

        results = speedtest_servers(servers=custom_servers)

        print(f"Saving results to '{out_file}'")
        with open(out_file, 'a') as f:
            f.writelines(str(result.dict()) + "\n" for result in results)

        print(f"{datetime.datetime.now()}: {len(custom_servers)} tested in {time.perf_counter() - start_time:.3f} s")
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
