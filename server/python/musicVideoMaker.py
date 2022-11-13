from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import sys
import get_genre

vocal_path = sys.argv[1]
text = sys.argv[2]
genre = get_genre.get_genre(text)

video_name={
    "Folk":"reggae.mp4",
    "Rap":"trap.mp4",
    "Metal":"headbanger.mp4",
    "K-Pop":"kpop.mp4",
    "Disco":"gme.mp4",
    "Pop":"trap.mp4",
    "Funk":"rickroll.mp4",
    "Rock":"rock.mp4",
    "R&B":"trap.mp4",
    "Country":"idk.mp4",
    "Rap":"trap.mp4",
    "Jazz":"jazz.mp4",
    "Indie":"highschool.mp4"
}

def genre_to_video(genre):
    return video_name.get(genre, "marius.mp4")

# video_clip = VideoFileClip('/videos/{}'.format(genre_to_video))
genre_audio = AudioFileClip("./server/python/songs/{}.mp3".format(genre))
video_clip = VideoFileClip('./server/python/100_0001.mov')
vocal_audio = AudioFileClip(vocal_path)

genre_clip = genre_audio.volumex(0.5)
vocal_clip = vocal_audio.volumex(2.0)

final_audio = CompositeAudioClip([genre_clip, vocal_clip])

final_clip = video_clip.set_audio(final_audio)
final_clip.write_videofile("./server/python/output/{}".format(sys.argv[3]))

print(sys.argv[3])
sys.stdout.flush()