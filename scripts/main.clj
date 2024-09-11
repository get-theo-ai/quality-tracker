(ns main
  (:require [babashka.curl :as curl]
            [clojure.core.async :refer [<! >! go go-loop timeout]]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [datalevin.core :as d]
            [edamame.core :as edn]
            [jsonista.core :as json]))

(def schema (-> "schema.edn"
                slurp
                edn/parse-string))

(def conn (d/get-conn "db" schema {:auto-entity-time? true}))

(def test-set (-> "test-set.edn"
                  slurp
                  edn/parse-string))

(def base-url
  #_"http://localhost:8000"
  "http://theo-prediction-engine-dev.internal:8000")


(def !open-api-spec (atom nil))


(defn open-api-spec []
  (if (nil? @!open-api-spec)
    (let [spec (-> (str base-url "/openapi.json")
                   (curl/get {:accept :json
                              :as :json-string-keys})
                   :body
                   json/read-value)]
      (reset! !open-api-spec spec)
      spec)
    @!open-api-spec))


(defn prompt-ids []
  (let [prompt-schema (->> ["paths" "/predict-file-upload" "post" "parameters"]
                           (get-in (open-api-spec))
                           (filter #(= "prompt" (get % "name")))
                           first)]
    (-> prompt-schema
        (get-in ["schema" "enum"]))))

(defn llm-clients []
  ["CLAUDE" "GPT" "COHERE"])


(defn submit [{:keys [id file-paths category llm-client prompt]}]
  (let [{:keys [status body]}
        (-> (str base-url "/predict-file-upload")
            (curl/post {:accept :json
                        :as :json
                        :query-params {:prediction_identifier id
                                       :llm_client llm-client
                                       :category category
                                       :prompt prompt}
                        :headers {"Content-Type" "multipart/form-data"}
                        :throw false
                        :raw-args (->> file-paths
                                       (reduce (fn [c p]
                                                 (conj c "-F" (str "files=@" p ";type=application/pdf")))
                                               []))}))]
    (if (= 200 status)
      (-> body
          (json/read-value json/keyword-keys-object-mapper)
          (assoc :status :success))
      {:status :failure
       :failure body})))

(defn build-file-paths [path]
  (let [f (io/file path)]
    (if (.isFile f)
      [path]
      (->> f
           file-seq
           (filter (fn [^java.io.File f'] (.isFile f')))
           (map (fn [^java.io.File f'] (.getPath f')))))))

(defn not-tried-builder [model prompt]
  (fn [{:keys [id]}]
    (let [in-db (->> (d/q '[:find [(pull ?r [*]) ...]
                            :in $ ?id ?prompt ?model
                            :where
                            [?r :run/id ?id]
                            [?r :run/prompt-version ?prompt]
                            [?r :run/model ?model]]
                          @conn id prompt model)
                     first)]
      (nil? in-db))))

(defn weird-prob-builder [model prompt]
  (fn [{:keys [id]}]
    (let [in-db (->> (d/q '[:find [(pull ?r [*]) ...]
                            :in $ ?id ?prompt ?model
                            :where
                            [?r :run/id ?id]
                            [?r :run/prompt-version ?prompt]
                            [?r :run/model ?model]
                            [?r :run/status :success]
                            (not [?r :run/prob-claimant-min "0%"])
                            (not [?r :run/prob-claimant-max "0%"])
                            (not [?r :run/prob-respondent-min "0%"])
                            (not [?r :run/prob-respondent-max "0%"])]
                          @conn id prompt model)
                     first)]
      (nil? in-db))))

(defn failed-builder [model prompt]
  (fn [{:keys [id]}]
    (let [in-db (->> (d/q '[:find [(pull ?r [*]) ...]
                            :in $ ?id ?prompt ?model
                            :where
                            [?r :run/id ?id]
                            [?r :run/prompt-version ?prompt]
                            [?r :run/model ?model]
                            [?r :run/status :success]]
                          @conn id prompt model))]
      (empty? in-db))))


(defn failed-or-not-tried-builder [model prompt]
  (let [pred-a (failed-builder model prompt)
        pred-b (not-tried-builder model prompt)]
    (fn [x]
      (or (pred-a x)
          (pred-b x)))))

(defn tmp-builder [model prompt-version]
  (fn [x] (= "2" (:id x))))

(def !should-run? (atom false))

(defn cancel-run []
  (reset! !should-run? false)
  :ok)


(defn success-txs [model prompt-version
                   {:keys [id category path case-name award] :as base}
                   {:keys [status
                           failure
                           prediction
                           input_tokens
                           output_tokens
                           prediction_prompt_version
                           prediction_LLM_model_temperature
                           LLM_model] :as resp}]
  [(cond-> {:run/id id
            :run/category category
            :run/file-path path
            :run/case-name case-name
            :run/award award
            :run/status status
            :run/response-str (str resp)
            :run/model model
            :run/prompt-version prompt-version}
     failure
     (assoc :run/failure-str failure)
     input_tokens
     (assoc :run/input-tokens input_tokens)
     output_tokens
     (assoc :run/output-tokens output_tokens)
     prediction_LLM_model_temperature
     (assoc :run/temperature prediction_LLM_model_temperature)
     prediction
     (assoc :run/prob-claimant-min
            (get-in prediction [:prob_of_winning_claimant :min] "0%")
            :run/prob-claimant-max
            (get-in prediction [:prob_of_winning_claimant :max] "0%")
            :run/award-claimant-min
            (get-in prediction [:award_claimant :min] 0)
            :run/award-claimant-max
            (get-in prediction [:award_claimant :max] 0)
            :run/prob-respondent-min
            (get-in prediction [:prob_of_winning_respondent :min] "0%")
            :run/prob-respondent-max
            (get-in prediction [:prob_of_winning_respondent :max] "0%")
            :run/award-respondent-min
            (get-in prediction [:award_respondent :min] 0)
            :run/award-respondent-max
            (get-in prediction [:award_respondent :max] 0)
            :run/confidence-min
            (get-in prediction [:confidence_range :min] "0%")
            :run/confidence-max
            (get-in prediction [:confidence_range :max] "0%")
            :run/case-summary
            (get prediction :case_summary "")
            :run/calculation-rationale
            (get prediction :calculation_rationale "")))])


(defn exception-txs [model prompt-version
                     {:keys [id category path case-name award] :as base}
                     ex]
  [{:run/id id
    :run/category category
    :run/file-path path
    :run/case-name case-name
    :run/award award
    :run/status :failure
    :run/failure-str (str ex)
    :run/model model
    :run/prompt-version prompt-version}])


(defn run
  ([]
   (run (fn [& _] (constantly true))))
  ([pred-builder]
   (reset! !should-run? true)
   (go
     (println "Starting!")
     (doseq [model (llm-clients)]
       (println "For model" model)
       (doseq [prompt-version (prompt-ids)]
         (println "For prompt-version" prompt-version)
         (let [pred (pred-builder model prompt-version)]
           (doseq [{:keys [id path category case-name award] :as base}
                   (->> test-set
                        (filter pred))]
             (when @!should-run?
               (try
                 (println "\nRunning for:")
                 (println "- id:" id)
                 (println "- prompt-version:" prompt-version)
                 (println "- model:" model)
                 (println "- path:" path)
                 (let [file-paths (build-file-paths path)
                       {:keys [status] :as resp} (submit {:id id
                                                          :category category
                                                          :file-paths file-paths
                                                          :llm-client model
                                                          :prompt prompt-version})]
                   (println "Status for id" id "is" status)
                   (d/transact! conn (success-txs model prompt-version base resp)))
                 (catch Throwable ex
                   (d/transact! conn (exception-txs model prompt-version base ex)))))))))
     (println "\n\nDone!"))))


(defn big-loop [builder]
  (dotimes [_ 50]
    (run builder)
    (Thread/sleep 15000)))

(defn forever-loop [builder]
  (while true
    (run builder)
    (Thread/sleep 15000)))


(defn percentage [s]
  (->> s
       (re-find #"\d+")
       Integer/parseInt))

(defn mean [min max]
  (/ (+ min max) 2))

(defn stdev [min max]
  (if (= 0 (- max min))
    (do
      (println "!!!!!!!!")
      Long/MAX_VALUE)
    (/ (- max min)
       (* 2 1.65))))

(defn z-score [target m std]
  (/ (- target m)
     std))

(defn prediction-rating [z]
  (let [z' (abs z)]
    (cond
      (< z' 1) "A+"
      (< z' 2) "A"
      (< z' 3) "B"
      (< z' 4) "C"
      (< z' 5) "D"
      :else "F")))

(defn spread-rating [m std]
  (let [x (/ std (if (= 0 m) Long/MAX_VALUE m))]
    (cond
      (< x 0.1) "A+"
      (< x 0.15) "A"
      (< x 0.20) "B"
      (< x 0.3) "C"
      (< x 0.4) "D"
      :else "F")))

(defn query-ids [db ids]
  (->> ids
       (d/q '[:find [(pull ?r [*]) ...]
              :in $ [?ids ...]
              :where
              [?r :run/id ?ids]]
            db)))

(defn query-id [db id]
  (->> [id]
       (query-ids db)))

(defn query-status [db status]
  (->> status
       (d/q '[:find [(pull ?r [*]) ...]
              :in $ ?status
              :where
              [?r :run/status ?status]]
            db)))

(defn query-by-tuple [db id prompt model]
  (->> (d/q '[:find [(pull ?r [*]) ...]
              :in $ ?id ?prompt ?model
              :where
              [?r :run/id ?id]
              [?r :run/prompt-version ?prompt]
              [?r :run/model ?model]]
            db id prompt model)))

(defn close! []
  (d/close conn))

(defn dump-to-csv [file-name headers data-set]
  (with-open [writer (io/writer file-name)]
    (csv/write-csv writer
                   (concat headers data-set))))

(defn csv-mapper [{:keys [run/id run/model run/prompt-version
                          run/file-path run/case-name
                          run/award
                          run/prob-claimant-min run/prob-claimant-max
                          run/prob-respondent-min run/prob-respondent-max
                          run/award-claimant-min run/award-claimant-max
                          run/award-respondent-min run/award-respondent-max
                          run/case-summary run/calculation-rationale]}]
  (let [perc-claimant (percentage prob-claimant-max)
        perc-respondent (percentage prob-respondent-max)
        [a-min a-max] (if (>= perc-claimant perc-respondent)
                        [award-claimant-min award-claimant-max]
                        [award-respondent-min award-respondent-max])
        m (mean a-min a-max)
        std (stdev a-min a-max)
        p-rating (-> award
                     (z-score m std)
                     prediction-rating)
        s-rating (spread-rating m std)]
    [id case-name model prompt-version
     award p-rating s-rating
     prob-claimant-min prob-claimant-max
     award-claimant-min award-claimant-max
     prob-respondent-min prob-respondent-max
     award-respondent-min award-respondent-max
     case-summary calculation-rationale]))


(def csv-header [["id" "case-name" "model" "prompt-version"
                  "real-award" "prediction-rating" "spread-rating"
                  "prob-claimant-min" "prob-claimant-max"
                  "award-claimant-min" "award-claimant-max"
                  "prob-respondent-min" "prob-respondent-max"
                  "award-respondent-min" "award-respondent-max"
                  "case-summary" "calculation-rationale"]])

(comment

  #_(->> (d/q '[:find ?r :where [?r :run/id]]
              @conn)
         count)
  
  #_(->> (d/q '[:find [(pull ?r [*]) ...]
                :in $
                :where
                [?r :run/status :success]
                (or (and [?r :run/prob-respondent-min "0%"]
                         [?r :run/prob-respondent-max "0%"])
                    (and [?r :run/prob-claimant-min "0%"]
                         [?r :run/prob-claimant-max "0%"]))]
              @conn)
         #_(take 2)
         (map csv-mapper)
         (dump-to-csv "out.csv" csv-header))


  
  (->> (d/q '[:find [(pull ?r [*]) ...]
              :in $
              :where
              [?r :run/status :success]
              (not [?r :run/prob-claimant-min "0%"]
                   [?r :run/prob-claimant-max "0%"])
              (not [?r :run/prob-respondent-min "0%"]
                   [?r :run/prob-respondent-max "0%"])
              (or (not [?r :run/award-claimant-min 0]
                       [?r :run/award-claimant-max 0])
                  (not [?r :run/award-respondent-min 0]
                       [?r :run/award-respondent-max 0]))]
            @conn)
       #_count
       #_(take 2)
       (mapv csv-mapper)
       (dump-to-csv "out.csv" csv-header))

  #_{:shared 304

     :zeroed-out 707

     :success 1011
     :failure 8472
     :total 9483}

  )
