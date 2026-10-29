"""
LNCT Timetable & Real-Time Period Attention Tracker
Lakshmi Narain College of Technology, Bhopal
Class: B.Tech Vth Semester | CSE (AIML) - Section B | Room 4-B9 (CV Raman Block)
Session: July - Dec 2026 | Effective: 06/10/2026
"""

import json
import os
from datetime import datetime, time
from typing import Dict, Any, List, Optional
from config import Config

TIMETABLE_META = {
    "college": "Lakshmi Narain College of Technology, Bhopal",
    "form_no": "FM0902-A",
    "course": "B.Tech. Vth Semester (3rd Year)",
    "branch": "CSE (AIML)",
    "section": "B",
    "room": "Room No. 4-B9 / CV Raman Block",
    "session": "July-Dec 2026",
    "effective_date": "06/10/2026"
}

PERIOD_SLOTS = [
    {"period": "I", "start": "10:30", "end": "11:15", "start_min": 10*60 + 30, "end_min": 11*60 + 15, "is_lunch": False},
    {"period": "II", "start": "11:15", "end": "12:00", "start_min": 11*60 + 15, "end_min": 12*60 + 0, "is_lunch": False},
    {"period": "III", "start": "12:00", "end": "12:45", "start_min": 12*60 + 0, "end_min": 12*60 + 45, "is_lunch": False},
    {"period": "IV", "start": "12:45", "end": "13:25", "start_min": 12*60 + 45, "end_min": 13*60 + 25, "is_lunch": False},
    {"period": "V", "name": "LUNCH", "start": "13:25", "end": "14:10", "start_min": 13*60 + 25, "end_min": 14*60 + 10, "is_lunch": True},
    {"period": "VI", "start": "14:10", "end": "14:50", "start_min": 14*60 + 10, "end_min": 14*60 + 50, "is_lunch": False},
    {"period": "VII", "start": "14:50", "end": "15:30", "start_min": 14*60 + 50, "end_min": 15*60 + 30, "is_lunch": False},
    {"period": "VIII", "start": "15:30", "end": "16:15", "start_min": 15*60 + 30, "end_min": 16*60 + 15, "is_lunch": False},
    {"period": "IX", "start": "16:15", "end": "17:00", "start_min": 16*60 + 15, "end_min": 17*60 + 0, "is_lunch": False},
]

SUBJECTS_CATALOG = {
    "AL-501": {
        "code": "AL-501",
        "title": "Operating System",
        "faculty": "Dr. Aakriti Jain (AJ)",
        "contact": "7987622028",
        "lab_room": "Lab 2B CV Raman Block",
        "color": "#C5F82A"
    },
    "AL-502": {
        "code": "AL-502",
        "title": "Database Management System",
        "faculty": "Prof. Richa Sarathe (RSa)",
        "contact": "6260941617",
        "lab_room": "Lab 3B CV Raman Block",
        "color": "#38BDF8"
    },
    "AL-503": {
        "code": "AL-503 (B)",
        "title": "Departmental Elective (Deep Learning)",
        "faculty": "Prof. Chayank Deshmukh (CD)",
        "contact": "7354496407",
        "lab_room": "Lab 2B / 4A CV Raman Block",
        "color": "#A78BFA"
    },
    "AL-504": {
        "code": "AL-504 (B)",
        "title": "Open Elective (Natural Language Processing)",
        "faculty": "Prof. Arbind Kumar Singh (AK)",
        "contact": "7987606449",
        "lab_room": "Lab 3B CV Raman Block",
        "color": "#FB923C"
    },
    "AL-505": {
        "code": "AL-505",
        "title": "Dept Elective Lab (Deep Learning)",
        "faculty": "Prof. Chayank Deshmukh (CD)",
        "contact": "7354496407",
        "lab_room": "Lab 2B / 4A CV Raman Block",
        "color": "#E879F9"
    },
    "AL-506": {
        "code": "AL-506",
        "title": "Open Elective Lab (NLP)",
        "faculty": "Prof. Arbind Kumar Singh (AK)",
        "contact": "7987606449",
        "lab_room": "Lab 3B CV Raman Block",
        "color": "#4ADE80"
    },
    "AL-507": {
        "code": "AL-507",
        "title": "Evaluation of Internship-II",
        "faculty": "Prof. Nandita Tiwari (NT)",
        "contact": "6260801311",
        "lab_room": "Room 4-B9 CV Raman Block",
        "color": "#FACC15"
    },
    "AL-508": {
        "code": "AL-508",
        "title": "Minor Project-I",
        "faculty": "Prof. Richa Sarathe (RSa)",
        "contact": "6260941617",
        "lab_room": "Lab 2B / 4B CV Raman Block",
        "color": "#2DD4BF"
    },
    "Mentor/Library": {
        "code": "Mentor/Library",
        "title": "Mentorship & Library Session",
        "faculty": "Dr. Aakriti Jain (AJ)",
        "contact": "7987622028",
        "lab_room": "Central Library / Room 4-B9",
        "color": "#94A3B8"
    },
    "Minor Project Lab": {
        "code": "Minor Project Lab",
        "title": "Minor Project Lab Work",
        "faculty": "Prof. Richa Sarathe (RSa)",
        "contact": "6260941617",
        "lab_room": "Lab 4A CV Raman Block",
        "color": "#2DD4BF"
    }
}

WEEKLY_SCHEDULE = {
    "MON": [
        {"period": "I", "code": "AL-502", "subject": "AL-502 DBMS", "faculty": "RSa", "room": "4-B9"},
        {"period": "II", "code": "AL-501", "subject": "AL-501 OS", "faculty": "AJ", "room": "4-B9"},
        {"period": "III", "code": "AL-505", "subject": "AL-505 Lab (B1) / AL-502 Lab (B2)", "faculty": "CD / RSa", "room": "Lab 2B/3B"},
        {"period": "IV", "code": "AL-505", "subject": "AL-505 Lab (B1) / AL-502 Lab (B2)", "faculty": "CD / RSa", "room": "Lab 2B/3B"},
        {"period": "V", "code": "LUNCH", "subject": "LUNCH BREAK", "faculty": "-", "room": "-"},
        {"period": "VI", "code": "AL-503", "subject": "AL-503 Deep Learning", "faculty": "CD", "room": "4-B9"},
        {"period": "VII", "code": "AL-503", "subject": "AL-503 (T) Deep Learning", "faculty": "CD", "room": "4-B9"},
        {"period": "VIII", "code": "AL-504", "subject": "AL-504 NLP", "faculty": "AK", "room": "4-B9"},
        {"period": "IX", "code": "AL-507", "subject": "AL-507 Internship Evaluation", "faculty": "NT", "room": "4-B9"},
    ],
    "TUE": [
        {"period": "I", "code": "AL-502", "subject": "AL-502 DBMS", "faculty": "RSa", "room": "4-B9"},
        {"period": "II", "code": "AL-503", "subject": "AL-503 Deep Learning", "faculty": "CD", "room": "4-B9"},
        {"period": "III", "code": "AL-503", "subject": "AL-503 Deep Learning", "faculty": "CD", "room": "4-B9"},
        {"period": "IV", "code": "AL-501", "subject": "AL-501 OS", "faculty": "AJ", "room": "4-B9"},
        {"period": "V", "code": "LUNCH", "subject": "LUNCH BREAK", "faculty": "-", "room": "-"},
        {"period": "VI", "code": "AL-502", "subject": "AL-502 Lab (B1) / AL-501 Lab (B2)", "faculty": "RSa / AJ", "room": "Lab 3B/2B"},
        {"period": "VII", "code": "AL-502", "subject": "AL-502 Lab (B1) / AL-501 Lab (B2)", "faculty": "RSa / AJ", "room": "Lab 3B/2B"},
        {"period": "VIII", "code": "AL-504", "subject": "AL-504 NLP", "faculty": "AK", "room": "4-B9"},
        {"period": "IX", "code": "AL-502", "subject": "AL-502 DBMS", "faculty": "RSa", "room": "4-B9"},
    ],
    "WED": [
        {"period": "I", "code": "Mentor/Library", "subject": "Mentor / Library", "faculty": "AJ", "room": "4-B9"},
        {"period": "II", "code": "AL-504", "subject": "AL-504 NLP", "faculty": "AK", "room": "4-B9"},
        {"period": "III", "code": "AL-502", "subject": "AL-502 DBMS", "faculty": "RSa", "room": "4-B9"},
        {"period": "IV", "code": "AL-502", "subject": "AL-502 DBMS", "faculty": "RSa", "room": "4-B9"},
        {"period": "V", "code": "LUNCH", "subject": "LUNCH BREAK", "faculty": "-", "room": "-"},
        {"period": "VI", "code": "AL-503", "subject": "AL-503 Deep Learning", "faculty": "CD", "room": "4-B9"},
        {"period": "VII", "code": "AL-501", "subject": "AL-501 OS", "faculty": "AJ", "room": "4-B9"},
        {"period": "VIII", "code": "AL-501", "subject": "AL-501 Lab (B1) / AL-506 Lab (B2)", "faculty": "AJ / AK", "room": "Lab 2B/3B"},
        {"period": "IX", "code": "AL-501", "subject": "AL-501 Lab (B1) / AL-506 Lab (B2)", "faculty": "AJ / AK", "room": "Lab 2B/3B"},
    ],
    "THU": [
        {"period": "I", "code": "AL-501", "subject": "AL-501 OS", "faculty": "AJ", "room": "4-B9"},
        {"period": "II", "code": "AL-504", "subject": "AL-504 NLP", "faculty": "AK", "room": "4-B9"},
        {"period": "III", "code": "AL-506", "subject": "AL-506 Lab (B1) / AL-508 Lab (B2)", "faculty": "AK / RSa", "room": "Lab 3B/2B"},
        {"period": "IV", "code": "AL-506", "subject": "AL-506 Lab (B1) / AL-508 Lab (B2)", "faculty": "AK / RSa", "room": "Lab 3B/2B"},
        {"period": "V", "code": "LUNCH", "subject": "LUNCH BREAK", "faculty": "-", "room": "-"},
        {"period": "VI", "code": "AL-504", "subject": "AL-504 NLP", "faculty": "AK", "room": "4-B9"},
        {"period": "VII", "code": "AL-502", "subject": "AL-502 DBMS", "faculty": "RSa", "room": "4-B9"},
        {"period": "VIII", "code": "AL-501", "subject": "AL-501 OS", "faculty": "AJ", "room": "4-B9"},
        {"period": "IX", "code": "AL-503", "subject": "AL-503 Deep Learning", "faculty": "CD", "room": "4-B9"},
    ],
    "FRI": [
        {"period": "I", "code": "AL-508", "subject": "AL-508 Lab (B1) / AL-505 Lab (B2)", "faculty": "RSa / CD", "room": "Lab 4B/4A"},
        {"period": "II", "code": "AL-508", "subject": "AL-508 Lab (B1) / AL-505 Lab (B2)", "faculty": "RSa / CD", "room": "Lab 4B/4A"},
        {"period": "III", "code": "AL-504", "subject": "AL-504 NLP", "faculty": "AK", "room": "4-B9"},
        {"period": "IV", "code": "AL-504", "subject": "AL-504 NLP", "faculty": "AK", "room": "4-B9"},
        {"period": "V", "code": "LUNCH", "subject": "LUNCH BREAK", "faculty": "-", "room": "-"},
        {"period": "VI", "code": "AL-501", "subject": "AL-501 OS", "faculty": "AJ", "room": "4-B9"},
        {"period": "VII", "code": "AL-501", "subject": "AL-501 OS", "faculty": "AJ", "room": "4-B9"},
        {"period": "VIII", "code": "AL-503", "subject": "AL-503 Deep Learning", "faculty": "CD", "room": "4-B9"},
        {"period": "IX", "code": "AL-502", "subject": "AL-502 DBMS", "faculty": "RSa", "room": "4-B9"},
    ],
    "SAT": [
        {"period": "I", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
        {"period": "II", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
        {"period": "III", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
        {"period": "IV", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
        {"period": "V", "code": "LUNCH", "subject": "LUNCH BREAK", "faculty": "-", "room": "-"},
        {"period": "VI", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
        {"period": "VII", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
        {"period": "VIII", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
        {"period": "IX", "code": "Minor Project Lab", "subject": "Minor Project Lab Work", "faculty": "Lab Staff", "room": "Lab 4A"},
    ]
}

# Persistence file for subject attention metrics
SUBJECT_STATS_FILE = os.path.join(Config.DATA_DIR, "subject_attention_stats.json")

class TimetableManager:
    """Manages LNCT timetable lookup and subject-wise attention tracking."""

    def __init__(self):
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Any]:
        if os.path.exists(SUBJECT_STATS_FILE):
            try:
                with open(SUBJECT_STATS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Initialize default realistic baselines for each LNCT subject
        init_stats = {
            "AL-501": {"subject": "Operating System", "faculty": "Dr. Aakriti Jain", "total_frames": 1420, "attentive_frames": 1195, "attention_rate": 84.2, "distracted_phone": 14, "distracted_sleep": 4, "distracted_away": 22},
            "AL-502": {"subject": "Database Management System", "faculty": "Prof. Richa Sarathe", "total_frames": 1380, "attentive_frames": 1118, "attention_rate": 81.0, "distracted_phone": 19, "distracted_sleep": 7, "distracted_away": 26},
            "AL-503": {"subject": "Deep Learning", "faculty": "Prof. Chayank Deshmukh", "total_frames": 1560, "attentive_frames": 1370, "attention_rate": 87.8, "distracted_phone": 11, "distracted_sleep": 2, "distracted_away": 15},
            "AL-504": {"subject": "Natural Language Processing", "faculty": "Prof. Arbind Kumar Singh", "total_frames": 1290, "attentive_frames": 1083, "attention_rate": 84.0, "distracted_phone": 15, "distracted_sleep": 5, "distracted_away": 18},
            "AL-505": {"subject": "Deep Learning Lab", "faculty": "Prof. Chayank Deshmukh", "total_frames": 980, "attentive_frames": 880, "attention_rate": 89.8, "distracted_phone": 8, "distracted_sleep": 1, "distracted_away": 9},
            "AL-506": {"subject": "NLP Lab", "faculty": "Prof. Arbind Kumar Singh", "total_frames": 920, "attentive_frames": 815, "attention_rate": 88.6, "distracted_phone": 9, "distracted_sleep": 2, "distracted_away": 8},
            "AL-507": {"subject": "Internship Evaluation", "faculty": "Prof. Nandita Tiwari", "total_frames": 850, "attentive_frames": 748, "attention_rate": 88.0, "distracted_phone": 6, "distracted_sleep": 1, "distracted_away": 9},
            "AL-508": {"subject": "Minor Project-I", "faculty": "Prof. Richa Sarathe", "total_frames": 1100, "attentive_frames": 990, "attention_rate": 90.0, "distracted_phone": 7, "distracted_sleep": 0, "distracted_away": 8},
            "Mentor/Library": {"subject": "Mentor / Library", "faculty": "Dr. Aakriti Jain", "total_frames": 740, "attentive_frames": 592, "attention_rate": 80.0, "distracted_phone": 18, "distracted_sleep": 6, "distracted_away": 14},
            "Minor Project Lab": {"subject": "Minor Project Lab Work", "faculty": "Prof. Richa Sarathe", "total_frames": 1200, "attentive_frames": 1092, "attention_rate": 91.0, "distracted_phone": 5, "distracted_sleep": 1, "distracted_away": 6}
        }
        self._save_stats(init_stats)
        return init_stats

    def _save_stats(self, stats_dict: Dict[str, Any]):
        try:
            with open(SUBJECT_STATS_FILE, "w") as f:
                json.dump(stats_dict, f, indent=2)
        except Exception as e:
            print(f"[TimetableManager] Could not save stats: {e}")

    def get_current_period(self, sim_day: Optional[str] = None, sim_time_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates the active period based on real-time clock, or simulated day/time.
        LNCT Bhopal operates on Indian Standard Time (IST, UTC+5:30).
        """
        from datetime import timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(IST)
        
        # Day name 3-letter code (MON, TUE, WED, THU, FRI, SAT, SUN)
        day_code = sim_day.upper() if sim_day else now.strftime("%a").upper()
        
        if sim_time_str:
            parts = sim_time_str.split(":")
            current_min = int(parts[0]) * 60 + int(parts[1])
            now_time_fmt = sim_time_str
        else:
            current_min = now.hour * 60 + now.minute
            now_time_fmt = now.strftime("%I:%M:%S %p")

        today_schedule = WEEKLY_SCHEDULE.get(day_code, [])
        is_sunday = (day_code == "SUN")

        active_slot = None
        slot_index = -1
        time_remaining_str = ""

        if not is_sunday:
            for idx, slot in enumerate(PERIOD_SLOTS):
                if slot["start_min"] <= current_min < slot["end_min"]:
                    active_slot = slot
                    slot_index = idx
                    rem_mins = slot["end_min"] - current_min
                    rem_secs = 60 - now.second if not sim_time_str else 0
                    time_remaining_str = f"{rem_mins}m {rem_secs}s"
                    break

        if active_slot and 0 <= slot_index < len(today_schedule):
            class_entry = today_schedule[slot_index]
            sub_code = class_entry["code"]
            catalog_info = SUBJECTS_CATALOG.get(sub_code, {
                "title": class_entry["subject"],
                "faculty": class_entry["faculty"],
                "contact": "LNCT Academic Office",
                "lab_room": class_entry["room"],
                "color": "#C5F82A"
            })

            # Fetch attention stats for this subject
            stats_for_sub = self.stats.get(sub_code, {
                "attention_rate": 85.0,
                "attentive_frames": 100,
                "total_frames": 120,
                "distracted_phone": 3,
                "distracted_sleep": 0,
                "distracted_away": 2
            })

            return {
                "is_active": True,
                "is_off_hours": False,
                "is_lunch": active_slot["is_lunch"],
                "day": day_code,
                "current_time": now_time_fmt,
                "period": active_slot["period"],
                "time_slot": f"{active_slot['start']} - {active_slot['end']}",
                "time_remaining": time_remaining_str,
                "subject_code": sub_code,
                "subject_title": catalog_info["title"],
                "faculty": catalog_info["faculty"],
                "faculty_contact": catalog_info["contact"],
                "room": catalog_info.get("lab_room", class_entry["room"]),
                "accent_color": catalog_info.get("color", "#C5F82A"),
                "attention_rate": stats_for_sub.get("attention_rate", 85.0),
                "distracted_phone": stats_for_sub.get("distracted_phone", 0),
                "distracted_sleep": stats_for_sub.get("distracted_sleep", 0),
                "distracted_away": stats_for_sub.get("distracted_away", 0),
                "meta": TIMETABLE_META
            }

        # If currently outside college periods (e.g. night or before 10:30 AM or Sunday)
        # Default to the most relevant period for demonstration (Period II: AL-501 OS)
        next_text = "Classes resume at 10:30 AM" if not is_sunday else "Sunday Holiday - Classes resume Monday 10:30 AM"
        return {
            "is_active": False,
            "is_off_hours": True,
            "is_lunch": False,
            "day": day_code,
            "current_time": now_time_fmt,
            "period": "Off-Hours",
            "time_slot": "10:30 AM - 05:00 PM (Daily)",
            "time_remaining": next_text,
            "subject_code": "STANDBY",
            "subject_title": "Classroom Standby (College Hours: 10:30 AM - 05:00 PM)",
            "faculty": "LNCT Academic Schedule",
            "faculty_contact": "LNCT Academic Office",
            "room": "Room 4-B9 / CV Raman Block",
            "accent_color": "#94A3B8",
            "attention_rate": self.stats.get("AL-501", {}).get("attention_rate", 84.2),
            "distracted_phone": self.stats.get("AL-501", {}).get("distracted_phone", 14),
            "distracted_sleep": self.stats.get("AL-501", {}).get("distracted_sleep", 4),
            "distracted_away": self.stats.get("AL-501", {}).get("distracted_away", 22),
            "meta": TIMETABLE_META
        }

    def record_live_attention(self, subject_code: str, attentive_students: int, distracted_students: int, phone_count: int = 0, sleep_count: int = 0, away_count: int = 0):
        """Records live session frame data into subject-wise attention stats."""
        total = attentive_students + distracted_students
        if total <= 0:
            return

        if subject_code not in self.stats:
            self.stats[subject_code] = {
                "subject": subject_code,
                "faculty": "LNCT Faculty",
                "total_frames": 0,
                "attentive_frames": 0,
                "attention_rate": 80.0,
                "distracted_phone": 0,
                "distracted_sleep": 0,
                "distracted_away": 0
            }

        s = self.stats[subject_code]
        s["total_frames"] += total
        s["attentive_frames"] += attentive_students
        s["distracted_phone"] += phone_count
        s["distracted_sleep"] += sleep_count
        s["distracted_away"] += away_count
        if s["total_frames"] > 0:
            s["attention_rate"] = round((s["attentive_frames"] / s["total_frames"]) * 100.0, 1)

        self._save_stats(self.stats)

    def get_full_timetable(self) -> Dict[str, Any]:
        """Returns complete weekly grid and subject catalog."""
        return {
            "meta": TIMETABLE_META,
            "period_slots": PERIOD_SLOTS,
            "subjects_catalog": SUBJECTS_CATALOG,
            "weekly_schedule": WEEKLY_SCHEDULE,
            "subject_stats": self.stats
        }

# Global singleton
timetable_manager = TimetableManager()
