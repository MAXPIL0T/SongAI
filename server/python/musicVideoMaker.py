from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import sys
import get_genre

vocal_path = sys.argv[1]
text = sys.argv[2]
genre = get_genre.get_genre(text)

# video_clip = VideoFileClip('/videos/{}.mp4'.format(genre))
# genre_audio = AudioFileClip("/beats/{}.mp3".format(genre))
video_clip = VideoFileClip('./server/python/video.mp4')
genre_audio = AudioFileClip('./server/python/audio.wav')
vocal_audio = AudioFileClip(vocal_path)

# genre_clip = genre_audio.volume(0.75)
# vocal_clip = vocal_audio.volume(1.0)

final_audio = CompositeAudioClip([genre_audio, vocal_audio])

final_clip = video_clip.set_audio(final_audio)
final_clip.write_videofile("./server/python/output/{}".format(sys.argv[3]))

print("python/output/{}".format(sys.argv[3]))
sys.stdout.flush()