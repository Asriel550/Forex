"""One-time acquisition of official HistData M1 ZIPs, without extraction."""
from pathlib import Path
from html.parser import HTMLParser
import argparse,hashlib,json,time,zipfile,datetime
import requests
from filelock import FileLock
class Forms(HTMLParser):
    def __init__(self):super().__init__();self.forms=[];self.current=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form':self.current={'action':a.get('action'), 'fields':{}};self.forms.append(self.current)
        elif tag=='input' and self.current is not None and a.get('name'):
            self.current['fields'][a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.current=None

def acquire(raw_dir,year,month=None):
    period=str(year)+(f'{month:02d}' if month else '')
    url='https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/'+str(year)+(f'/{month}' if month else '')
    path=raw_dir/f'HISTDATA_COM_ASCII_EURUSD_M1_{period}.zip'
    raw_dir.mkdir(parents=True, exist_ok=True)
    # El lock cubre comprobación, descarga, sustitución atómica y lectura.
    # Dos jobs/procesos nunca escriben el mismo .part simultáneamente.
    with FileLock(str(path) + '.lock', timeout=600):
        if path.exists() and zipfile.is_zipfile(path):
            print(f'{period}: verified local ZIP',flush=True)
        else:
            with requests.Session() as s:
                s.headers['User-Agent']='Forex-HistData-archive/1.0 (sequential historical research downloads)'
                for attempt in range(3):
                    try:
                        r=s.get(url,timeout=(45,90));r.raise_for_status();break
                    except (requests.ConnectionError,requests.Timeout):
                        if attempt==2:raise
                        time.sleep(3*(attempt+1))
                p=Forms();p.feed(r.text)
                f=next((x for x in p.forms if x['fields'].get('timeframe')=='M1' and x['fields'].get('fxpair')=='EURUSD' and x['fields'].get('tk')),None)
                if f is None:raise RuntimeError(f'{period}: HistData did not publish a download form')
                r=s.post('https://www.histdata.com/get.php',data=f['fields'],headers={'Referer':url,'Origin':'https://www.histdata.com'},timeout=(15,120));r.raise_for_status()
                tmp=path.with_suffix('.part');tmp.write_bytes(r.content)
                if not zipfile.is_zipfile(tmp):raise RuntimeError(f'{period}: server returned a non-ZIP response')
                with zipfile.ZipFile(tmp) as z:
                    if z.testzip():raise RuntimeError('ZIP checksum failure')
                tmp.replace(path)
            time.sleep(2)
        with zipfile.ZipFile(path) as z:
            entries=[dict(name=i.filename,uncompressed_bytes=i.file_size) for i in z.infolist()]
    return dict(period=period,provider='HistData.com',source_url=url,zip_file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,entries=entries)
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=root/'data/histdata_raw')
    parser.add_argument('--manifest',type=Path,default=root/'data/histdata_sources.json')
    parser.add_argument('--start-year',type=int,default=2021)
    parser.add_argument('--end-year',type=int,default=2025)
    parser.add_argument('--through-month',default='2026-10',help='YYYY-MM; optional partial following year')
    args=parser.parse_args()
    if not 2000<=args.start_year<=args.end_year<datetime.datetime.now(datetime.timezone.utc).year:
        parser.error('Use completed historical years only')
    periods=[(y,None) for y in range(args.start_year,args.end_year+1)]
    if args.through_month:
        limit=datetime.datetime.strptime(args.through_month,'%Y-%m')
        now=datetime.datetime.now(datetime.timezone.utc)
        if limit.year!=args.end_year+1 or (limit.year,limit.month)>=(now.year,now.month):
            parser.error('through-month must be a completed month in the following year')
        periods += [(limit.year,m) for m in range(1,limit.month+1)]
    args.raw_dir.mkdir(parents=True,exist_ok=True);args.manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for year,month in periods:
        item=acquire(args.raw_dir,year,month);manifest.append(item)
        args.manifest.write_text(json.dumps(dict(acquired_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),archives=manifest),indent=2))
        print({k:item[k] for k in ('period','bytes','sha256')},flush=True)
