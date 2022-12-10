import os
import time
import spotipy
import tqdm
import config
import json
import lyricsgenius as lg
import logging
from threading import Thread

logging.basicConfig(filename=config.DOWNLOAD_LOG, filemode='a+', format='%(name)s - %(levelname)s - %(message)s')
# Authentication - without user
client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=config.CLIENT_ID,
                                                              client_secret=config.CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

remake = False
testing = False
res = {}


def scrape_lyrics(artist_name, song_name):  # function to get lyrics from genius.com
    genius = lg.Genius(config.GENIUS_TOKEN)
    genius.verbose = False
    retries = 0
    while retries < 3:
        try:
            song = genius.search_song(song_name, artist_name, get_full_info=False)
        except Exception as e:
            time.sleep(5)
            logging.error(e)
            retries += 1
            continue
        if song is not None:
            text = list(song.lyrics.split('\n'))
            text1 = []
            for line in text:
                if '[' not in line and line:
                    text1.append(line)
            if not text1:
                return 'no text'
            if 'Lyrics' in text1[0]:
                text1[0] = text1[0][text1[0].find('Lyrics') + 6:]
            if 'Embed' in text1[-1]:
                text1[-1] = text1[-1][:-5]
                while text1[-1] and text1[-1][-1].isdigit():
                    text1[-1] = text1[-1][:-1]
            return '\n'.join(text1)
        else:
            return 'no text'
    return 'no text'


def get_uris():
    if not remake:
        return
    # not testing
    used = set()
    with open(config.URIS_FILE, 'w', encoding='utf-8') as fw:
        for filename in tqdm.tqdm(os.listdir("../../data")):
            with open(os.path.join("../../data", filename), 'r') as f:
                data = json.load(f)
            for playlist in data['playlists']:
                for track in playlist['tracks']:
                    uri = track['track_uri']
                    if uri not in used:
                        used.add(uri)
                        fw.write(uri + config.SEP_LOC + track['artist_name']
                                 + config.SEP_LOC + track['track_name'] + config.SEP_GLOB)
    print(len(used))


def get_song_data(track_uri: int):
    retries = 0
    _res = None
    while retries < 3:
        try:
            _res = sp.audio_features(track_uri)[0]
            break
        except Exception as e:
            logging.error(e)
            time.sleep(5)
            retries += 1
    if _res is None:
        return {}
    data = {
        'danceability': _res['danceability'],
        'energy': _res['energy'],
        'key': _res['key'],
        'loudness': _res['loudness'],
        'mode': _res['mode'],
        'valence': _res['valence'],
        'tempo': _res['tempo'],
    }
    return data


def download_data(track: str):
    global res
    uri, artist, song = track.split(config.SEP_LOC)
    lyrics = scrape_lyrics(artist, song)
    if lyrics == "no text":
        return
    meta = get_song_data(uri)
    if len(meta) == 0:
        return
    # encoded = list(map(lambda x: x.item(), model.encode(lyrics)))
    res['data'].append({"lyrics": lyrics, "meta": meta, "artist": artist, "song": song})


def gen_db():
    global res
    # uris are unique!

    if testing:
        fr = "test_uris.txt"
    else:
        fr = config.URIS_FILE
    if testing:
        fw = "test_db.json"
    else:
        fw = config.SONG_DATA_FILE
    with open(config.DOWNLOAD_LAST, 'r') as f:
        start, index = map(int, f.readline().split())
    with open(fr, 'r', encoding='utf-8') as f:
        data = f.readline().split(config.SEP_GLOB)
        res = {"data": []}
        step = 100
        stime = time.perf_counter()
        for i in tqdm.trange(start, len(data), step):
            threads = [Thread(target=download_data, args=(data[i + j],)) for j in range(step)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if len(res['data']) >= 1000:
                logging.info(f"Finished {i = }, {index = }.")
                with open(config.DOWNLOAD_LAST, 'w') as fmem:
                    fmem.write(f"{i + step} {index + 1}")
                print(f"\nAlready a thousand! Finished in {time.perf_counter() - stime} s; next i will be {i + step}.")
                stime = time.perf_counter()
                with open(fw.format(index), 'w', encoding='utf-8') as fout:
                    json.dump(res, fout)
                index += 1
                res = {"data": []}


def test(fr: str, index: int):
    with open(fr, 'r', encoding='utf-8') as f:
        data = f.readline().split(config.SEP_GLOB)
        track = data[index]
        uri, artist, song = track.split(config.SEP_LOC)
        meta = get_song_data(uri)
        print(meta)
        lyrics = scrape_lyrics(artist, song)
        print(lyrics)


if __name__ == '__main__':
    print(os.getcwd())
    gen_db()
    # test("uris.txt", 8557 + 576 * 100 - 2)
