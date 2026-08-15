from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import pandas as pd
from PIL import Image, ImageTk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\cow")

IMAGE_DIR = PROJECT_ROOT / "dataset"

LABEL_FILE = (
    PROJECT_ROOT
    / "bcs"
    / "labels"
    / "bcs_labeling_dataset.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

BCS_VALUES = [
    1.00, 1.25, 1.50, 1.75,
    2.00, 2.25, 2.50, 2.75,
    3.00, 3.25, 3.50, 3.75,
    4.00, 4.25, 4.50, 4.75,
    5.00,
]

IMAGE_MAX_WIDTH = 900
IMAGE_MAX_HEIGHT = 430

WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 850


# ============================================================
# APPLICATION
# ============================================================

class BCSLabelingTool:

    def __init__(self, root: tk.Tk):

        self.root = root

        self.root.title(
            "COW PLF - BCS Reference Labeling Tool"
        )

        self.root.geometry(
            f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
        )

        self.root.minsize(
            1050,
            700,
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close,
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.current_index = None

        self.current_photo = None

        self.selected_bcs = None

        self.assessor_id = ""

        self.status_message = tk.StringVar(
            value="Enter Assessor ID to begin."
        )

        self.selected_bcs_text = tk.StringVar(
            value="No BCS selected"
        )

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        self.load_dataset()

        # ----------------------------------------------------
        # Build UI
        # ----------------------------------------------------

        self.build_ui()

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        self.find_first_unlabeled()

    # ========================================================
    # DATASET
    # ========================================================

    def load_dataset(self):

        if not LABEL_FILE.exists():

            raise FileNotFoundError(
                f"\nBCS labeling dataset not found:\n"
                f"{LABEL_FILE}\n\n"
                f"Run create_labeling_dataset.py first."
            )

        self.df = pd.read_csv(
            LABEL_FILE,
            dtype=str,
        )

        required_columns = [
            "cow_id",
            "image_name",
            "bcs_score",
            "bcs_source",
            "assessor_id",
            "assessment_notes",
            "label_status",
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns in BCS dataset:\n{missing}"
            )

        # Normalize status values.
        self.df["label_status"] = (
            self.df["label_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):

        # ----------------------------------------------------
        # Main container
        # ----------------------------------------------------

        outer = tk.Frame(
            self.root
        )

        outer.pack(
            fill="both",
            expand=True,
        )

        # ----------------------------------------------------
        # Canvas + scrollbar
        # ----------------------------------------------------

        self.canvas = tk.Canvas(
            outer,
            highlightthickness=0,
        )

        scrollbar = tk.Scrollbar(
            outer,
            orient="vertical",
            command=self.canvas.yview,
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.canvas.configure(
            yscrollcommand=scrollbar.set
        )

        self.content = tk.Frame(
            self.canvas,
            padx=20,
            pady=15,
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )

        self.content.bind(
            "<Configure>",
            self.update_scroll_region,
        )

        self.canvas.bind(
            "<Configure>",
            self.resize_canvas_window,
        )

        # Mouse wheel.
        self.canvas.bind_all(
            "<MouseWheel>",
            self.on_mousewheel,
        )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = tk.Frame(
            self.content
        )

        header.pack(
            fill="x",
            pady=(0, 10),
        )

        tk.Label(
            header,
            text="COW PLF - BCS REFERENCE LABELING",
            font=("Arial", 21, "bold"),
        ).pack()

        tk.Label(
            header,
            text=(
                "Independent visual assessment only. "
                "Do NOT use weight or morphometric measurements."
            ),
            font=("Arial", 11),
        ).pack(
            pady=(5, 0)
        )

        # ----------------------------------------------------
        # ASSESSOR
        # ----------------------------------------------------

        assessor_frame = tk.Frame(
            self.content,
            relief="groove",
            borderwidth=1,
            padx=12,
            pady=10,
        )

        assessor_frame.pack(
            fill="x",
            pady=(0, 10),
        )

        tk.Label(
            assessor_frame,
            text="Assessor ID:",
            font=("Arial", 12, "bold"),
        ).pack(
            side="left"
        )

        self.assessor_entry = tk.Entry(
            assessor_frame,
            width=22,
            font=("Arial", 12),
        )

        self.assessor_entry.pack(
            side="left",
            padx=10,
        )

        self.assessor_entry.bind(
            "<Return>",
            self.set_assessor,
        )

        tk.Button(
            assessor_frame,
            text="SET ASSESSOR",
            width=16,
            height=1,
            font=("Arial", 10, "bold"),
            command=self.set_assessor,
        ).pack(
            side="left"
        )

        self.assessor_status = tk.Label(
            assessor_frame,
            text="Not set",
            font=("Arial", 11, "bold"),
        )

        self.assessor_status.pack(
            side="left",
            padx=15,
        )

        # ----------------------------------------------------
        # COW INFORMATION
        # ----------------------------------------------------

        info_frame = tk.Frame(
            self.content
        )

        info_frame.pack(
            fill="x",
            pady=5,
        )

        self.cow_label = tk.Label(
            info_frame,
            text="Cow ID: -",
            font=("Arial", 17, "bold"),
        )

        self.cow_label.pack(
            side="left"
        )

        self.image_label = tk.Label(
            info_frame,
            text="Image: -",
            font=("Arial", 12),
        )

        self.image_label.pack(
            side="right"
        )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        self.progress_label = tk.Label(
            self.content,
            text="",
            font=("Arial", 11, "bold"),
        )

        self.progress_label.pack(
            pady=5
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_frame = tk.Frame(
            self.content,
            relief="sunken",
            borderwidth=2,
            bg="white",
            height=450,
        )

        image_frame.pack(
            fill="x",
            pady=10,
        )

        image_frame.pack_propagate(
            False
        )

        self.photo_label = tk.Label(
            image_frame,
            text="Loading image...",
            bg="white",
            font=("Arial", 12),
        )

        self.photo_label.pack(
            expand=True
        )

        # ----------------------------------------------------
        # BCS TITLE
        # ----------------------------------------------------

        tk.Label(
            self.content,
            text="Select Reference BCS",
            font=("Arial", 15, "bold"),
        ).pack(
            pady=(10, 5)
        )

        tk.Label(
            self.content,
            text=(
                "Choose the score based ONLY on visible body-condition "
                "characteristics."
            ),
            font=("Arial", 10),
        ).pack()

        # ----------------------------------------------------
        # BCS BUTTONS
        # ----------------------------------------------------

        bcs_frame = tk.Frame(
            self.content
        )

        bcs_frame.pack(
            pady=8
        )

        self.bcs_buttons = {}

        for index, score in enumerate(BCS_VALUES):

            button = tk.Button(
                bcs_frame,
                text=f"{score:.2f}",
                width=7,
                height=2,
                font=("Arial", 10, "bold"),
                command=lambda s=score: self.select_bcs(s),
            )

            button.grid(
                row=index // 9,
                column=index % 9,
                padx=3,
                pady=3,
            )

            self.bcs_buttons[
                float(score)
            ] = button

        # ----------------------------------------------------
        # SELECTED BCS DISPLAY
        # ----------------------------------------------------

        selected_frame = tk.Frame(
            self.content,
            relief="groove",
            borderwidth=1,
            padx=15,
            pady=8,
        )

        selected_frame.pack(
            fill="x",
            pady=5,
        )

        tk.Label(
            selected_frame,
            text="Current selection:",
            font=("Arial", 11, "bold"),
        ).pack(
            side="left"
        )

        self.selected_label = tk.Label(
            selected_frame,
            textvariable=self.selected_bcs_text,
            font=("Arial", 13, "bold"),
        )

        self.selected_label.pack(
            side="left",
            padx=10,
        )

        # ----------------------------------------------------
        # REVIEW
        # ----------------------------------------------------

        self.review_button = tk.Button(
            self.content,
            text="REVIEW / INSUFFICIENT VISUAL INFORMATION",
            width=48,
            height=2,
            font=("Arial", 11, "bold"),
            command=self.select_review,
        )

        self.review_button.pack(
            pady=8
        )

        # ----------------------------------------------------
        # NOTES
        # ----------------------------------------------------

        notes_frame = tk.Frame(
            self.content
        )

        notes_frame.pack(
            fill="x",
            pady=5,
        )

        tk.Label(
            notes_frame,
            text="Assessment notes (optional):",
            font=("Arial", 10, "bold"),
        ).pack(
            anchor="w"
        )

        self.notes_entry = tk.Text(
            notes_frame,
            height=3,
            font=("Arial", 10),
            wrap="word",
        )

        self.notes_entry.pack(
            fill="x",
            pady=3,
        )

        tk.Label(
            notes_frame,
            text=(
                "Example: visible ribs and prominent hip bones. "
                "Leave blank if no additional note is needed."
            ),
            font=("Arial", 9),
        ).pack(
            anchor="w"
        )

        # ----------------------------------------------------
        # NAVIGATION
        # ----------------------------------------------------

        navigation = tk.Frame(
            self.content
        )

        navigation.pack(
            fill="x",
            pady=12,
        )

        self.previous_button = tk.Button(
            navigation,
            text="← PREVIOUS",
            width=16,
            height=2,
            font=("Arial", 10, "bold"),
            command=self.previous_cow,
        )

        self.previous_button.pack(
            side="left"
        )

        self.save_button = tk.Button(
            navigation,
            text="SAVE & NEXT →",
            width=22,
            height=2,
            font=("Arial", 12, "bold"),
            command=self.save_current,
        )

        self.save_button.pack(
            side="right"
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status_frame = tk.Frame(
            self.content
        )

        status_frame.pack(
            fill="x",
            pady=(0, 15),
        )

        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_message,
            font=("Arial", 10),
            anchor="w",
            justify="left",
        )

        self.status_label.pack(
            fill="x"
        )

    # ========================================================
    # CANVAS
    # ========================================================

    def update_scroll_region(self, event=None):

        self.canvas.configure(
            scrollregion=self.canvas.bbox("all")
        )

    def resize_canvas_window(self, event):

        self.canvas.itemconfig(
            self.canvas_window,
            width=event.width,
        )

    def on_mousewheel(self, event):

        self.canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units",
        )

    # ========================================================
    # ASSESSOR
    # ========================================================

    def set_assessor(self, event=None):

        assessor = (
            self.assessor_entry
            .get()
            .strip()
        )

        if not assessor:

            messagebox.showwarning(
                "Assessor ID Required",
                "Please enter an assessor ID.\n\n"
                "Example: A_01",
            )

            self.assessor_entry.focus_set()

            return

        self.assessor_id = assessor

        self.assessor_status.config(
            text=f"Active: {assessor}"
        )

        self.status_message.set(
            f"Assessor {assessor} set. "
            f"Now visually assess the cow and select a BCS."
        )

        self.assessor_entry.config(
            state="disabled"
        )

    # ========================================================
    # PROGRESS
    # ========================================================

    def get_progress(self):

        total = len(self.df)

        completed = (
            self.df["label_status"]
            .astype(str)
            .str.upper()
            .isin(
                [
                    "LABELED",
                    "REVIEW",
                ]
            )
            .sum()
        )

        return int(completed), int(total)

    # ========================================================
    # FIND FIRST UNLABELED
    # ========================================================

    def find_first_unlabeled(self):

        for index in range(len(self.df)):

            status = (
                str(
                    self.df.loc[
                        index,
                        "label_status",
                    ]
                )
                .strip()
                .upper()
            )

            if status == "UNLABELED":

                self.current_index = index

                self.load_current()

                return

        self.show_complete()

    # ========================================================
    # LOAD CURRENT COW
    # ========================================================

    def load_current(self):

        if self.current_index is None:
            return

        row = self.df.loc[
            self.current_index
        ]

        cow_id = str(
            row["cow_id"]
        ).strip()

        image_name = str(
            row["image_name"]
        ).strip()

        self.cow_label.config(
            text=f"Cow ID: {cow_id}"
        )

        self.image_label.config(
            text=f"Image: {image_name}"
        )

        completed, total = self.get_progress()

        self.progress_label.config(
            text=(
                f"Progress: {completed} / {total} processed"
                f"     |     "
                f"Row {self.current_index + 1} / {total}"
            )
        )

        # ----------------------------------------------------
        # Reset selection
        # ----------------------------------------------------

        self.selected_bcs = None

        self.selected_bcs_text.set(
            "No BCS selected"
        )

        self.reset_bcs_buttons()

        # ----------------------------------------------------
        # Notes
        # ----------------------------------------------------

        self.notes_entry.delete(
            "1.0",
            tk.END,
        )

        existing_notes = row.get(
            "assessment_notes",
            "",
        )

        if pd.notna(existing_notes):

            existing_notes = str(
                existing_notes
            ).strip()

            if existing_notes:

                self.notes_entry.insert(
                    "1.0",
                    existing_notes,
                )

        # ----------------------------------------------------
        # Assessor
        # ----------------------------------------------------

        existing_assessor = row.get(
            "assessor_id",
            "",
        )

        if (
            not self.assessor_id
            and pd.notna(existing_assessor)
            and str(existing_assessor).strip()
        ):

            self.assessor_entry.config(
                state="normal"
            )

            self.assessor_entry.delete(
                0,
                tk.END,
            )

            self.assessor_entry.insert(
                0,
                str(existing_assessor).strip(),
            )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_path = IMAGE_DIR / image_name

        if not image_path.exists():

            self.photo_label.config(
                image="",
                text=(
                    "IMAGE NOT FOUND\n\n"
                    f"{image_path}"
                ),
            )

            self.current_photo = None

            self.status_message.set(
                "Image not found. You may mark REVIEW."
            )

            return

        try:

            image = Image.open(
                image_path
            )

            image.thumbnail(
                (
                    IMAGE_MAX_WIDTH,
                    IMAGE_MAX_HEIGHT,
                ),
                Image.Resampling.LANCZOS,
            )

            self.current_photo = ImageTk.PhotoImage(
                image
            )

            self.photo_label.config(
                image=self.current_photo,
                text="",
            )

        except Exception as exc:

            self.photo_label.config(
                image="",
                text=(
                    "Unable to load image:\n"
                    f"{exc}"
                ),
            )

            self.current_photo = None

        self.status_message.set(
            "Visually assess the cow, select a BCS or REVIEW, "
            "then click SAVE & NEXT."
        )

    # ========================================================
    # BCS SELECTION
    # ========================================================

    def select_bcs(self, score):

        self.selected_bcs = float(score)

        self.selected_bcs_text.set(
            f"BCS {score:.2f} SELECTED"
        )

        self.reset_bcs_buttons()

        button = self.bcs_buttons[
            float(score)
        ]

        button.config(
            relief="sunken",
            borderwidth=4,
        )

        self.status_message.set(
            f"BCS {score:.2f} selected. "
            f"Click SAVE & NEXT to save this assessment."
        )

    # ========================================================
    # REVIEW
    # ========================================================

    def select_review(self):

        self.selected_bcs = "REVIEW"

        self.selected_bcs_text.set(
            "REVIEW SELECTED"
        )

        self.reset_bcs_buttons()

        self.review_button.config(
            relief="sunken",
            borderwidth=4,
        )

        self.status_message.set(
            "REVIEW selected. Add a note explaining why "
            "the image is insufficient if possible."
        )

    # ========================================================
    # RESET BUTTONS
    # ========================================================

    def reset_bcs_buttons(self):

        for button in self.bcs_buttons.values():

            button.config(
                relief="raised",
                borderwidth=2,
            )

        self.review_button.config(
            relief="raised",
            borderwidth=2,
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save_current(self):

        if self.current_index is None:

            messagebox.showinfo(
                "Complete",
                "All cows have already been processed.",
            )

            return

        # ----------------------------------------------------
        # Assessor validation
        # ----------------------------------------------------

        assessor = (
            self.assessor_entry
            .get()
            .strip()
        )

        if not assessor:

            messagebox.showwarning(
                "Assessor ID Required",
                "Enter an Assessor ID first.\n\n"
                "Example:\n"
                "A_01",
            )

            self.assessor_entry.config(
                state="normal"
            )

            self.assessor_entry.focus_set()

            return

        self.assessor_id = assessor

        # ----------------------------------------------------
        # BCS validation
        # ----------------------------------------------------

        if self.selected_bcs is None:

            messagebox.showwarning(
                "BCS Not Selected",
                "Please select a BCS score or REVIEW "
                "before saving.",
            )

            return

        # ----------------------------------------------------
        # Notes
        # ----------------------------------------------------

        notes = (
            self.notes_entry
            .get(
                "1.0",
                tk.END,
            )
            .strip()
        )

        # ----------------------------------------------------
        # Current cow
        # ----------------------------------------------------

        cow_id = str(
            self.df.loc[
                self.current_index,
                "cow_id",
            ]
        ).strip()

        image_name = str(
            self.df.loc[
                self.current_index,
                "image_name",
            ]
        ).strip()

        # ----------------------------------------------------
        # Save BCS
        # ----------------------------------------------------

        if self.selected_bcs != "REVIEW":

            score = float(
                self.selected_bcs
            )

            # Validate BCS.
            if score < 1.0 or score > 5.0:

                messagebox.showerror(
                    "Invalid BCS",
                    "BCS must be between 1.00 and 5.00.",
                )

                return

            # Validate 0.25 increments.
            quarter_value = round(
                score * 4
            )

            if abs(
                score * 4 - quarter_value
            ) > 1e-9:

                messagebox.showerror(
                    "Invalid BCS",
                    "BCS must use 0.25 increments.",
                )

                return

            self.df.at[
                self.current_index,
                "bcs_score",
            ] = f"{score:.2f}"

            self.df.at[
                self.current_index,
                "bcs_source",
            ] = "trained_assessor"

            self.df.at[
                self.current_index,
                "assessor_id",
            ] = assessor

            self.df.at[
                self.current_index,
                "assessment_notes",
            ] = notes

            self.df.at[
                self.current_index,
                "label_status",
            ] = "LABELED"

            result_text = (
                f"Cow {cow_id} saved successfully.\n\n"
                f"BCS: {score:.2f}\n"
                f"Assessor: {assessor}"
            )

        # ----------------------------------------------------
        # Save REVIEW
        # ----------------------------------------------------

        else:

            self.df.at[
                self.current_index,
                "bcs_score",
            ] = ""

            self.df.at[
                self.current_index,
                "bcs_source",
            ] = "trained_assessor"

            self.df.at[
                self.current_index,
                "assessor_id",
            ] = assessor

            self.df.at[
                self.current_index,
                "assessment_notes",
            ] = notes

            self.df.at[
                self.current_index,
                "label_status",
            ] = "REVIEW"

            result_text = (
                f"Cow {cow_id} saved as REVIEW.\n\n"
                f"Assessor: {assessor}"
            )

        # ----------------------------------------------------
        # WRITE CSV IMMEDIATELY
        # ----------------------------------------------------

        try:

            self.df.to_csv(
                LABEL_FILE,
                index=False,
            )

        except Exception as exc:

            messagebox.showerror(
                "SAVE FAILED",
                "The CSV could not be written.\n\n"
                f"Error:\n{exc}",
            )

            return

        # ----------------------------------------------------
        # Confirm save
        # ----------------------------------------------------

        self.status_message.set(
            result_text.replace(
                "\n",
                " | ",
            )
        )

        # ----------------------------------------------------
        # Move next
        # ----------------------------------------------------

        self.next_unlabeled()

    # ========================================================
    # NEXT
    # ========================================================

    def next_unlabeled(self):

        if self.current_index is None:
            return

        start = self.current_index + 1

        # Search forward.
        for index in range(
            start,
            len(self.df),
        ):

            status = (
                str(
                    self.df.loc[
                        index,
                        "label_status",
                    ]
                )
                .strip()
                .upper()
            )

            if status == "UNLABELED":

                self.current_index = index

                self.load_current()

                return

        # Search from beginning.
        for index in range(
            0,
            start,
        ):

            status = (
                str(
                    self.df.loc[
                        index,
                        "label_status",
                    ]
                )
                .strip()
                .upper()
            )

            if status == "UNLABELED":

                self.current_index = index

                self.load_current()

                return

        # Nothing left.
        self.show_complete()

    # ========================================================
    # PREVIOUS
    # ========================================================

    def previous_cow(self):

        if self.current_index is None:
            return

        if self.current_index <= 0:

            messagebox.showinfo(
                "First Cow",
                "This is the first cow in the dataset.",
            )

            return

        self.current_index -= 1

        self.load_current()

        self.status_message.set(
            "Previous cow loaded. "
            "Use SAVE & NEXT if you want to overwrite its label."
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    def show_complete(self):

        completed, total = self.get_progress()

        self.current_index = None

        self.progress_label.config(
            text=(
                f"COMPLETE: {completed} / {total} processed"
            )
        )

        self.cow_label.config(
            text="All cows processed"
        )

        self.image_label.config(
            text=""
        )

        self.photo_label.config(
            image="",
            text="BCS LABELING COMPLETE",
        )

        self.current_photo = None

        self.status_message.set(
            "All cows have been processed."
        )

        messagebox.showinfo(
            "BCS Labeling Complete",
            f"All {total} cows have been processed.\n\n"
            f"Labels saved to:\n{LABEL_FILE}",
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def on_close(self):

        answer = messagebox.askyesno(
            "Exit BCS Labeling Tool",
            "Exit the BCS labeling tool?\n\n"
            "All saved labels have already been written "
            "to the CSV.",
        )

        if answer:

            self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("COW PLF - BCS REFERENCE LABELING TOOL")
    print("=" * 80)
    print()
    print(f"Dataset : {LABEL_FILE}")
    print(f"Images  : {IMAGE_DIR}")
    print()
    print("Enter your Assessor ID.")
    print("Then visually assess each cow.")
    print("Do NOT use weight or morphometric measurements.")
    print()

    root = tk.Tk()

    BCSLabelingTool(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()