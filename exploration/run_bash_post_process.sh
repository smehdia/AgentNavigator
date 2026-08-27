#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Use app_navigation env with deps installed; ignore ~/.local packages.
export PYTHONNOUSERSITE=1
if [ -x "/home/mehdi/anaconda3/envs/app_navigation/bin/python" ]; then
  PYTHON="/home/mehdi/anaconda3/envs/app_navigation/bin/python"
elif [ -x "${CONDA_PREFIX:-}/bin/python" ] && [[ "${CONDA_DEFAULT_ENV:-}" == "app_navigation" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -x "/home/mehdi/miniforge3/envs/app_navigation/bin/python" ]; then
  PYTHON="/home/mehdi/miniforge3/envs/app_navigation/bin/python"
else
  PYTHON="python3"
fi
echo "Using Python: $PYTHON"

run_post_process() {
  local cfg="$1"
  echo "[post_process] START $cfg"
  if "$PYTHON" post_process.py --config "$cfg"; then
    echo "[post_process] SUCCESS $cfg"
  else
    echo "[post_process] FAILED $cfg (continuing)" >&2
  fi
}

# Post-process apps missing JSON artifacts (55 apps; continue on failure)
run_post_process configs/bloomberg_android.yaml
run_post_process configs/bolt_android.yaml
run_post_process configs/books_harmony.yaml
run_post_process configs/breaking_news_android.yaml
run_post_process configs/calendar_harmony.yaml
run_post_process configs/ciba_harmony.yaml
run_post_process configs/ctrip_harmony.yaml
run_post_process configs/didi_harmony.yaml
run_post_process configs/dongquidi_news_harmony.yaml
run_post_process configs/dropbox_harmony.yaml
run_post_process configs/duckduckgo_android.yaml
run_post_process configs/ebooks_reader_android.yaml
run_post_process configs/es_file_explorer_harmony.yaml
run_post_process configs/fenbi_harmony.yaml
run_post_process configs/game_center_harmony.yaml
run_post_process configs/google_keepnotes_android.yaml
run_post_process configs/home_workout_android.yaml
run_post_process configs/huawei_browser_harmony.yaml
run_post_process configs/iCurrency_harmony.yaml
run_post_process configs/jd_harmony.yaml
run_post_process configs/jiakao_baodian_harmony.yaml
run_post_process configs/kindle_android.yaml
run_post_process configs/kobo_books_android.yaml
run_post_process configs/kwai_harmony.yaml
run_post_process configs/meitu_harmony.yaml
run_post_process configs/myexpenses_android.yaml
run_post_process configs/notepad_harmony.yaml
run_post_process configs/opera_android.yaml
run_post_process configs/picture_this_harmony.yaml
run_post_process configs/road_road_pass_harmony.yaml
run_post_process configs/smartnews_android.yaml
run_post_process configs/soul_harmony.yaml
run_post_process configs/ssense_android.yaml
run_post_process configs/supercook_android.yaml
run_post_process configs/todo_huawei_harmony.yaml
run_post_process configs/tomato_todo_harmony.yaml
run_post_process configs/tonghuashun_harmony.yaml
run_post_process configs/vlc_harmony.yaml
run_post_process configs/vmall_harmony.yaml
run_post_process configs/wallet_harmony.yaml
run_post_process configs/weather_harmony.yaml
run_post_process configs/weibo_harmony.yaml
run_post_process configs/wikihow_android.yaml
run_post_process configs/x_android.yaml
run_post_process configs/xiao_yuan_ai_harmony.yaml
run_post_process configs/xiaohongshu_harmony.yaml
run_post_process configs/youdao_dictionary_harmony.yaml
# Train localizer for all apps (continue on failure)
"$PYTHON" train_localizer.py --app_dir explored_apps/accuweather --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/accuweather (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/agoda --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/agoda (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/airbnb --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/airbnb (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/alibaba_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/alibaba_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/aliexpress --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/aliexpress (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/amazon --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/amazon (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/anjuke_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/anjuke_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/ap_news --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/ap_news (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/app_gallary_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/app_gallary_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/autoscout24 --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/autoscout24 (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/babbel --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/babbel (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/baidu_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/baidu_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/basic_note --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/basic_note (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/battery_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/battery_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/bbc_sport --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/bbc_sport (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/bilibili_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/bilibili_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/bing --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/bing (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/bloomberg --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/bloomberg (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/bolt --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/bolt (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/books_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/books_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/breaking_news --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/breaking_news (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/broccoli --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/broccoli (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/calendar --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/calendar (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/calendar_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/calendar_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/chrome --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/chrome (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/ciba_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/ciba_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/clock --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/clock (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/cnn --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/cnn (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/coursera --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/coursera (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/ctrip_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/ctrip_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/didi_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/didi_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/dongquidi_news_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/dongquidi_news_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/dropbox_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/dropbox_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/duckduckgo --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/duckduckgo (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/ebay --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/ebay (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/ebooks_reader --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/ebooks_reader (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/es_file_explorer_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/es_file_explorer_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/fenbi_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/fenbi_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/game_center_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/game_center_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/google_keepnotes --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/google_keepnotes (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/google_maps --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/google_maps (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/google_playstore --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/google_playstore (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/home_workout --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/home_workout (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/huawei_browser_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/huawei_browser_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/iCurrency_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/iCurrency_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/imdb --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/imdb (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/instagram --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/instagram (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/jd_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/jd_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/jiakao_baodian_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/jiakao_baodian_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/joplin --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/joplin (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/kindle --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/kindle (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/kobo_books --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/kobo_books (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/kwai_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/kwai_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/linkedin --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/linkedin (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/markor --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/markor (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/meitu_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/meitu_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/myexpenses --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/myexpenses (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/notepad_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/notepad_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/omio --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/omio (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/onedrive --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/onedrive (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/opentable --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/opentable (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/opera --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/opera (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/osmand --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/osmand (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/outlook --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/outlook (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/outlook_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/outlook_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/picture_this_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/picture_this_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/pinterest --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/pinterest (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/retro_music --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/retro_music (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/road_road_pass_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/road_road_pass_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/settings --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/settings (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/simple_calendar_pro --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/simple_calendar_pro (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/smartnews --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/smartnews (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/soul_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/soul_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/sound_cloud --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/sound_cloud (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/spotify --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/spotify (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/ssense --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/ssense (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/starbucks --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/starbucks (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/supercook --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/supercook (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/target --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/target (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/temu --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/temu (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/tiktok --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/tiktok (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/todo_huawei_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/todo_huawei_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/tomato_todo_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/tomato_todo_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/tonghuashun_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/tonghuashun_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/trip_com --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/trip_com (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/tripadvisor --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/tripadvisor (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/uber --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/uber (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/uber_eats --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/uber_eats (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/vlc_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/vlc_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/vmall_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/vmall_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/wallet_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/wallet_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/walmart --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/walmart (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/weather_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/weather_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/weibo_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/weibo_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/wikihow --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/wikihow (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/wish --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/wish (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/x --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/x (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/xiao_yuan_ai_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/xiao_yuan_ai_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/xiaohongshu_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/xiaohongshu_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/yahoo_finance --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/yahoo_finance (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/yelp --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/yelp (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/youdao_dictionary_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/youdao_dictionary_harmony (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/youtube --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/youtube (continuing)" >&2
"$PYTHON" train_localizer.py --app_dir explored_apps/zhixing_train_harmony --root_dir explored_apps || echo "WARNING: train_localizer failed for explored_apps/zhixing_train_harmony (continuing)" >&2
