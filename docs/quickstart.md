# Your first acquisition

This walks through a complete session with Micro-Manager's demo devices, so you need
no hardware and can't break anything.  The camera is simulated and shows a moving
pattern rather than a sample.  It takes about five minutes.

Before you start, [install the app and the device adapters](install.md).

## Start the app with the demo devices

Start the app.  It first asks what to open:

![The startup dialog, with Layout and Configuration
dropdowns](images/startup-dialog.png){ width="420" loading=lazy }

Leave **Layout** on *Default*, set **Configuration** to *Demo configuration*, and click
**Start**.

!!! tip

    To skip this dialog, start the app with `pymm run --demo-config`.

The app opens on the **Acquire** tab:

![The Acquire tab with Channels and MDA panels docked either side of the image
viewer](images/acquire.png){ loading=lazy }

From left to right: the **Channels** panel, the image viewer, and the **MDA** panel,
where you set up acquisitions.  Along the top is a toolbar, with Snap and Live on the
left.

## Take a picture

1. In the **Channels** panel, click the circle next to **DAPI** to select that
   channel.  The selected channel is marked in yellow.
2. Click the **Snap** button (the camera icon, top left).

The image appears in a **Preview** tab in the viewer.

Now try changing the exposure:

1. Change DAPI's **Exposure** to `50`, and press ++enter++.
2. Click **Snap** again.

The pattern gets brighter.  Each channel remembers its own exposure.

## Look at it live

Click the **Live** button (the video camera icon, next to Snap).  The image updates
continuously.  While it runs, try selecting a different channel or changing the
exposure: the changes apply straight away.  Click **Live** again to stop.

!!! tip "Getting back an earlier snap"

    Each snap replaces the last one in the Preview, but the last ten are kept.  Click
    the image-stack icon below the viewer to [get one back](viewing.md#getting-back-an-earlier-snap).

## Run an acquisition

An *MDA* (multi-dimensional acquisition) is a series of images across channels, time
points, z positions and stage positions.  Let's take two channels, at five z positions,
at three time points.

In the **MDA** panel, each kind of dimension is a section you turn on with its
checkbox.  Click a section's title to open it.

![The MDA panel with the Channels and Z Stack sections
open](images/mda-panel.png){ width="520" loading=lazy }

1. **Channels:** check the box and open the section.  Click the green **+** twice to
   add two rows, then set them to **DAPI** and **FITC**.
2. **Z Stack:** check the box and open the section.  Leave the mode on **Symmetric**.
   Set **Range** to `4` µm, then **Step size** to `1` µm, and **Slices** shows `5`.
3. **Time Series:** check the box and open the section.  Click the green **+** to add a
   row.  Leave **Interval** at `1 s`, and set **Loops** to `3`.

    A closed section shows a summary on its right, so you can see what's set without
    opening it.

4. Click **Run**, at the bottom right.

A new tab opens in the viewer, and fills in as the images arrive.  The status bar at the
bottom of the window shows the progress, and then **Acquisition complete**.

![The finished acquisition, with t and z sliders below the
image](images/acquisition.png){ loading=lazy }

## Explore the result

- Drag the **t** and **z** sliders below the image to move through the data.
- Uncheck **DAPI** or **FITC** below the sliders to hide a channel.
- Click **Auto** next to a channel to rescale its contrast.

[Viewing and saving images](viewing.md) covers the rest of the viewer's controls.

## Save it

The **Saving** section was off, so this acquisition is in memory only and would be lost
when you quit.  You can still save it afterwards: click the **Save** button (the floppy
disk icon below the image), and choose a name and a format.

To save while an acquisition runs instead, which is what you should do for real data,
turn on the [Saving section](acquiring.md#saving) *before* clicking **Run**.

## Where next?

- [Setting up your microscope](microscope.md): go from the demo devices to your own
  hardware.
- [Acquiring images](acquiring.md): stage positions, tiles, well plates, autofocus.
- [Finding your way around](interface.md): the other panels, and arranging the window.
