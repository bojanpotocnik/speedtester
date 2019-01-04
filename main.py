import datetime
import itertools
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional, Sequence, Dict, Tuple, Any, Union, Iterator
from xml.etree import ElementTree

import dateutil.parser
import speedtest

__author__ = "Bojan Potočnik"


# region Speedtest servers

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


# endregion Speedtest servers


@dataclass
class SpeedtestResult:
    @dataclass
    class Server:
        id: int
        name: str
        url: str
        url2: Optional[str]
        host: str
        country: str
        country_code: str
        sponsor: str
        latitude: float
        longitude: float
        distance: float

        # noinspection SpellCheckingInspection
        @classmethod
        def new(cls, jsn: Dict[str, Any]) -> 'SpeedtestResult.Server':
            return cls(
                id=int(jsn["id"]),
                name=jsn["name"],
                url=jsn["url"],
                url2=jsn.get("url", None),
                host=jsn["host"],
                country=jsn["country"],
                country_code=jsn["cc"],
                sponsor=jsn["sponsor"],
                latitude=float(jsn["lat"]),
                longitude=float(jsn["lon"]),
                distance=float(jsn["d"])
            )

    @dataclass
    class Client:
        ip: str
        latitude: float
        longitude: float
        country_code: str
        isp: str
        isp_rating: float
        isp_download_average: float
        isp_upload_average: float
        rating: float
        logged_in: bool  # True type yet unknown.

        # noinspection SpellCheckingInspection
        @classmethod
        def new(cls, jsn: Dict[str, Any]) -> 'SpeedtestResult.Client':
            # This variable is extracted here because the true "True" type hasn't been observed yet.
            if jsn["loggedin"] == "0":
                logged_in = False
            elif jsn["loggedin"] == "1":
                logged_in = True
            else:
                logged_in = jsn["loggedin"]

            return cls(
                ip=jsn["ip"],
                latitude=float(jsn["lat"]),
                longitude=float(jsn["lon"]),
                country_code=jsn["country"],
                isp=jsn["isp"],
                isp_rating=float(jsn["isprating"]),
                isp_download_average=float(jsn["ispdlavg"]),
                isp_upload_average=float(jsn["ispulavg"]),
                rating=float(jsn["rating"]),
                logged_in=logged_in,
            )

    # region Local test parameters
    iteration: int
    start_time: datetime.datetime
    end_time: datetime.datetime
    # endregion Local test parameters
    # region Speedtest results
    timestamp: datetime.datetime
    bytes_sent: int
    bytes_received: int
    share: Optional[str]  # This requires additional POST to generate PNG image.
    ping: int
    download: float
    upload: float
    server: Server
    client: Client

    # endregion Speedtest results

    @classmethod
    def from_json(cls, jsn: Union[str, Dict[str, Any]], iteration: int,
                  start_time: datetime.datetime, end_time: datetime.datetime = None) -> 'SpeedtestResult':
        if isinstance(jsn, str):
            jsn = json.loads(jsn)

        return cls(
            iteration=iteration,
            start_time=start_time.astimezone(),
            end_time=end_time.astimezone() if end_time else datetime.datetime.now().astimezone(),
            timestamp=dateutil.parser.parse(jsn["timestamp"]).astimezone(),
            bytes_sent=int(jsn["bytes_sent"]),
            bytes_received=int(jsn["bytes_received"]),
            share=jsn["share"],
            ping=int(jsn["ping"]),
            download=float(jsn["download"]),
            upload=float(jsn["upload"]),
            server=cls.Server.new(jsn["server"]) if jsn["server"] else None,
            client=cls.Client.new(jsn["client"]) if jsn["client"] else None,
        )

    @classmethod
    def from_result(cls, result: speedtest.SpeedtestResults, iteration: int,
                    start_time: datetime.datetime, end_time: datetime.datetime = None) -> 'SpeedtestResult':
        # noinspection PyProtectedMember
        return cls(
            iteration=iteration,
            start_time=start_time.astimezone(),
            end_time=end_time.astimezone() if end_time else datetime.datetime.now().astimezone(),
            timestamp=dateutil.parser.parse(result.timestamp).astimezone(),
            bytes_sent=result.bytes_sent,
            bytes_received=result.bytes_received,
            share=result._share,
            ping=result.ping,
            download=result.download,
            upload=result.upload,
            server=cls.Server.new(result.server) if result.server else None,
            client=cls.Client.new(result.client) if result.client else None,
        )

    @classmethod
    def new(cls, result: Union[str, Dict[str, Any], speedtest.SpeedtestResults], iteration: int,
            start_time: datetime.datetime, end_time: datetime.datetime = None) -> 'SpeedtestResult':
        if isinstance(result, speedtest.SpeedtestResults):
            return cls.from_result(result, iteration, start_time, end_time)
        else:
            return cls.from_json(result, iteration, start_time, end_time)


def speedtest_servers(st: Optional[speedtest.Speedtest], servers: Optional[Sequence[SpeedtestServer]], iteration: int) \
        -> Iterator[Tuple[speedtest.Speedtest, SpeedtestResult]]:
    """Test multiple servers."""
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
    for server in all_servers:
        print(f"{datetime.datetime.now()}: Testing server {server}")
        start_time = time.perf_counter()
        start_timestamp = datetime.datetime.now()
        # Set this server as the best server as only this server is tested.
        st._best = server
        # Test speeds
        st.download()
        st.upload()
        end_timestamp = datetime.datetime.now()
        # Add information which is otherwise added in the library test function.
        st.results.server = server

        result = SpeedtestResult.new(st.results, iteration, start_timestamp, end_timestamp)
        print(f"{datetime.datetime.now()}: Done in {time.perf_counter() - start_time:.3f} s: {result}")

        yield st, result


# region Data logging

class File:

    def __init__(self, path: str = "results.csv") -> None:
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        path = os.path.abspath(path)

        if not os.path.isfile(path):
            headers = []
            for f in fields(SpeedtestResult):
                if f.name in ("server", "client"):
                    headers.extend(f"{f.name}_{fc.name}" for fc in fields(f.type))
                else:
                    headers.append(f.name)

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                print(",".join(headers), end="", flush=True, file=f)

        self.path = path

    def write(self, result: SpeedtestResult) -> None:
        values = []
        for field_name, field_value in asdict(result).items():
            if field_name in ("server", "client"):
                values.extend(fc_value for fc_value in asdict(getattr(result, field_name)).values())
            else:
                values.append(field_value)
        # Convert values to str
        for i, v in enumerate(values):
            if isinstance(v, str):
                values[i] = f'"{v}"'
            elif isinstance(v, datetime.datetime):
                values[i] = v.isoformat()
            else:
                values[i] = str(v)
        # Append row to file
        with open(self.path, 'a') as f:
            print(file=f)
            print(",".join(values), end="", flush=True, file=f)


# endregion Data logging


def main(interval: int = 5) -> None:
    """
    :param interval: Measurement repeat interval (minutes). Multiple servers will be evenly time-split so that
                     each server has this constant interval.
    """
    servers = get_servers(countries=("Slovenia",), names=("Ljubljana",), from_cache=True)
    print(f"Testing to {len(servers)} servers every {interval} minutes:")
    for srv in servers:
        print(f" - {srv}")
    interval_offset = 60 if (len(servers) <= interval) else interval * 60 / len(servers)

    file = File()
    st: speedtest.Speedtest = None  # Speedtest instance will be reused.

    for iteration in itertools.count():
        print(f"{iteration}. {datetime.datetime.now()}: Testing {len(servers)} servers...")

        ts_start = datetime.datetime.now()
        # Round up to next minute, but do not ignore this time for interval
        ts_next = (ts_start.replace(microsecond=0)
                   + datetime.timedelta(minutes=interval)
                   - datetime.timedelta(seconds=60 - ts_start.second + 5))  # TODO: Problem if the second is 0.
        ts_start = ts_start.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        # Calculate in-interval time offset for each server
        ts_offsets = {
            ts_start + datetime.timedelta(seconds=i * interval_offset): srv for i, srv in enumerate(servers)
        }

        while ts_offsets:
            result = None
            for ts, srv in ts_offsets.items():
                if datetime.datetime.now() >= ts:
                    st, result = next(speedtest_servers(st, servers=[srv], iteration=iteration))
                    file.write(result)
                    # This timestamp/server must was tested and must not be checked again.
                    del ts_offsets[ts]
                    break  # Iteration cannot continue as dictionary was modified while iterating.
            if not result:
                # Nothing was done, don't consume CPU.
                time.sleep(0.1)

        print(f"{iteration}. {datetime.datetime.now()}: {len(servers)}"
              f" tested in {(datetime.datetime.now() - ts_start).total_seconds():.3f} s")

        wait_time = (ts_next - datetime.datetime.now()).total_seconds()
        if wait_time <= 0:
            warnings.warn(f"Interval of {interval} minutes is {-wait_time} s too short for all the servers!")
        else:
            print(f"Waiting for {wait_time} seconds...")
            while datetime.datetime.now() < ts_next:
                time.sleep(0.5)


def test_servers() -> None:
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


def test_csv() -> None:
    file = File()

    for i in range(10):
        now = datetime.datetime.now()
        # noinspection PyCallByClass
        result = SpeedtestResult(iteration=i,
                                 start_time=now,
                                 end_time=now + datetime.timedelta(seconds=23),
                                 timestamp=now + datetime.timedelta(seconds=1),
                                 bytes_sent=256,
                                 bytes_received=1024,
                                 share=None,
                                 ping=20 + i,
                                 download=11.2, upload=3.8,
                                 server=SpeedtestResult.Server(
                                     id=i,
                                     name="name!",
                                     url="url!",
                                     url2="url2!",
                                     host="host!",
                                     country="country!",
                                     country_code="country_code!",
                                     sponsor="sponsor!",
                                     latitude=11.22,
                                     longitude=33.4,
                                     distance=0.5678
                                 ),
                                 client=SpeedtestResult.Client(
                                     ip=f"192.168.1.{i}",
                                     latitude=12.34,
                                     longitude=56.78,
                                     country_code="CC",
                                     isp="ISP!",
                                     isp_rating=3.7,
                                     isp_download_average=9.8,
                                     isp_upload_average=2.1,
                                     rating=1.1,
                                     logged_in=False
                                 ))
        file.write(result)
        time.sleep(0.2)


if __name__ == "__main__":
    # test_servers()
    # test_csv()
    main()
