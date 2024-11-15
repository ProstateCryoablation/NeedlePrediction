import logging
import os
from typing import Annotated, Optional

import vtk, qt, ctk, slicer

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import numpy as np
import torch
from scipy.stats import chi2
import pandas as pd
from PyQt5.QtCore import QTimer, QCoreApplication
import time
import SimpleITK as sitk
import sitkUtils
from vtk.util import numpy_support


#
# Iceball_needle
#


class Iceball_needle(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Iceball_needle")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Eva Beek (Surgical Planning Laboratory)"]
        self.parent.helpText = _("""This module places targets within a volume and ensures they do not overlap with other targets.""")
        self.parent.acknowledgementText = _("""Original development by...""")

    def initializeParameterNode(self):
        self.parameterNode = slicer.vtkMRMLScriptedModuleNode()
        slicer.mrmlScene.AddNode(self.parameterNode)

#
# Register sample data sets in Sample Data module
#

class Iceball_needleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # 3D Volume Selector
        self.volumeSelectorLabel = qt.QLabel("Select the volume:")
        self.layout.addWidget(self.volumeSelectorLabel)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("Select the binary volume.")
        self.layout.addWidget(self.volumeSelector)
        
    
        # 3D Point Selection Button for First Target
        self.firstTargetButton = qt.QPushButton("Select First needle in 3D")
        self.firstTargetButton.toolTip = "Pick a point for the first needle insertion in 3D space."
        self.layout.addWidget(self.firstTargetButton)
        self.firstTargetButton.connect('clicked(bool)', self.onSelectFirstNeedle)

        self.markupsNodeSelector = slicer.qMRMLNodeComboBox()
        self.markupsNodeSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.markupsNodeSelector.selectNodeUponCreation = True
        self.markupsNodeSelector.addEnabled = False
        self.markupsNodeSelector.removeEnabled = False
        self.markupsNodeSelector.noneEnabled = False
        self.markupsNodeSelector.showHidden = False
        self.markupsNodeSelector.showChildNodeTypes = False
        self.markupsNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.markupsNodeSelector.setToolTip("Select a markups node for the target")
        self.layout.addWidget(self.markupsNodeSelector)
        
        #Button to select Excel file
        self.selectButton = qt.QPushButton("Select Excel File with historical miss data")
        self.selectButton.toolTip = "Choose an excel file with miss vectors to be used as confidence interval."
        self.layout.addWidget(self.selectButton)
        self.selectButton.clicked.connect(self.LoadingPoints)

        # Create a checkbox for enabling/disabling automation
        self.automationCheckbox = ctk.ctkCheckBox()
        self.automationCheckbox.text = "Determine Third target"
        self.layout.addWidget(self.automationCheckbox)
        # Connect the checkbox to a function
        self.automationCheckbox.connect('toggled(bool)', self.onAutomationToggled)
        
        # Calculate and Place Second Target Button
        self.secondTargetButton = qt.QPushButton("Calculate possbile Second Target")
        self.secondTargetButton.toolTip = "Automatically calculate the second target within the volume."
        self.layout.addWidget(self.secondTargetButton)
        self.secondTargetButton.connect('clicked(bool)', self.onPlaceSecondTarget)

        # 3D Volume Selector
        self.volumeSelectorLabel2 = qt.QLabel("Select volume to calculate AI iceball on:")
        self.layout.addWidget(self.volumeSelectorLabel2)

        self.volumeIceSelector = slicer.qMRMLNodeComboBox()
        self.volumeIceSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeIceSelector.selectNodeUponCreation = True
        self.volumeIceSelector.addEnabled = False
        self.volumeIceSelector.removeEnabled = False
        self.volumeIceSelector.noneEnabled = False
        self.volumeIceSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeIceSelector.setToolTip("Select the volume to calculate AI Iceball.")
        self.layout.addWidget(self.volumeIceSelector)
        
        # Calculate AI Iceball
        self.AIiceball = qt.QPushButton("Calculate AI Iceball")
        self.AIiceball.toolTip = "Calculate the AI generated iceball for the best point"
        self.layout.addWidget(self.AIiceball)
        self.AIiceball.connect('clicked(bool)', self.onAI_iceball)

        # Stop Point Selection Button
        self.stopButton = qt.QPushButton("Stop Point Placement")
        self.stopButton.toolTip = "Stop point placement."
        self.layout.addWidget(self.stopButton)
        self.stopButton.connect('clicked(bool)', self.onStopSelecting)
        
        self.timer = QTimer()
        self.startTime = None
         # Set the timer interval (milliseconds)
        
        # Initialize variables
        self.firstTargetRAS = None
        self.volumeNode = None
        self.placement = 'Second placement'

#%%
    """
    Part to select first needle and save it in a markuspnode with name 'Needle'
    """
        
    def onSelectFirstNeedle(self):
        self.volumeNode = self.volumeSelector.currentNode()
        if not self.volumeNode:
            slicer.util.errorDisplay("Please select a volume first.")
            return
        self.placeFirstNeedle()

    def placeFirstNeedle(self):
        # Ensure volume is valid
        if not self.volumeNode:
            slicer.util.errorDisplay("Please select a valid volume.")
            return

        # Place the first needle in 3D space
        self.markupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        self.markupsNode.SetName("Needle")
        slicer.modules.markups.logic().StartPlaceMode(1)


#%%
    
    def LoadingPoints(self):
        """
        Loads historical data and saves it as an array
        """
        # Open a file dialog to select an Excel file
        fileDialog = qt.QFileDialog()
        fileDialog.setNameFilter("Excel Files (*.xlsx *.xls)")
        fileDialog.setFileMode(qt.QFileDialog.ExistingFile)

        if fileDialog.exec_():
            selectedFiles = fileDialog.selectedFiles()
            filePath = selectedFiles[0]

        data = pd.read_excel(filePath)
        print(data.head())

        R_coords = data.iloc[:, 6]
        A_coords = data.iloc[:, 7]
        S_coords = data.iloc[:, 8]

        self.points = np.array(list(zip(R_coords,A_coords,S_coords)))
        
#%%
    def onAutomationToggled(self, checked):
        """
        Checks if there are two or three needles used
        """
        if checked:
            # Automation enabled logic
            print("Third placement")
            self.placement = 'Third placement'
        else:
            # Automation disabled logic
            print("Second placement")  
            self.placement = 'Second placement'
        
    def onStopSelecting(self):
        """
        Used to stop selecting a point
        """
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        interactionNode.SetCurrentInteractionMode(slicer.vtkMRMLInteractionNode.ViewTransform)
        slicer.util.infoDisplay("Point placement mode stopped.")
#%%
    """
    Part used to time a certain action
    """
    def startTimer(self):
        self.timer_interval = 5000 
        self.start_time = time.time()  # Record the start time
        self.timer.start(self.timer_interval)  # Start the timer

    def stopTimer(self):
        self.timer.stop()  # Stop the timer
        self.elapsed_time = time.time() - self.start_time
    def onTimerTimeout(self):
        # This method will be called every 'self.timer_interval' milliseconds
        slicer.util.messageBox("Timer triggered!")  # Example action
        
#%%        
    def showPopupMessage(self):
        """
        Used to let the user know that a certain task is being performed
        """
        # Create a message box
        msgBox = qt.QMessageBox()
    
        # Set the message box's content
        msgBox.setIcon(qt.QMessageBox.Information)  # Type of icon (Information, Warning, Critical, etc.)
        msgBox.setText("A task is being performed")
        msgBox.setInformativeText("Please wait until the task completes.")
        msgBox.setWindowTitle("Task in Progress")
        
        # Add buttons (optional)
        msgBox.setStandardButtons(qt.QMessageBox.Ok)  # Add an OK button
    
        # Show the message box (exec() makes it modal)
        msgBox.exec()
        
#%%     
    def onPlaceSecondTarget(self):
        """
        Calculates center, radius and eigenvector from historical miss data with function calculate_ellipsoid_properties
        Calculates points and fractions with function calculateSecondTarget
        Visualizes heatmap
        """
        self.showPopupMessage()
        self.startTimer()
        self.center, self.radii, self.eigenvector= self.calculate_ellipsoid_properties(self.points)
        points, fractions = self.calculateSecondTarget(self.volumeNode)
        print('points are calculated' )
        
#        self.visualizeTargets(self.volumeNode, points, fractions) #creates heatmap
        self.stopTimer()
        slicer.util.messageBox(f"Processing completed in {self.elapsed_time:.2f} seconds.")
#%%        
    def calculate_ellipsoid_properties(self, points, confidence_level=0.95):
        """
        Calculates center, radius and eigenvectors of a confidence ellipse based 
        on a set of points
        
        """
        # 1. Calculate the center (mean) of the data points     
        center = np.mean(points, axis=0)   
        # 2. Calculate the covariance matrix of the points
        cov_matrix = np.cov(points.T) 
        # 3. Perform eigenvalue decomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  
        # 4. Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort indices in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]  # Columns of eigenvectors are sorted

        # 4. Scale the axes to the specified confidence level (default is 95%)
        chi2_val = chi2.ppf(confidence_level, df=3)  # Confidence level in 3D   
        # The lengths of the ellipsoid axes are proportional to sqrt(eigenvalue * chi2_val)
        radii = np.sqrt(eigenvalues * chi2_val)

        return center, radii, eigenvectors
        
    def calculateSecondTarget(self, volumeNode):
        """
        1. Creates a set of points 1.5 cm from first needle insertion
        2. Calculates voxels for general 95% confidence ellipse in the direction of the first miss with function voxelize_ellipsoid
        3. For each point, recenters the 95% confidence ellipse to its point and calculates overlap between ellipse and tumor with function ellipsoid_and_tumor_overlap_percentage
        3. For each point, calculates iceball formation with function voxelize_ellipsoid, and calculates overlap  between ellipse and tumor with function ellipsoid_and_tumor_overlap_percentage
        4. Optional could visualize the formed 95% confidence interval and iceball
        5. Normalizes both overlap scores
        6. Combines 95% confidence and iceball with tumor overlap scores into one final value, where weighting factors W1 and W2 could be adjusted
        """
        #determine first needle location
        markupsNodeTarget = slicer.util.getNode('Needle')
        firstNeedle = [0.0, 0.0, 0.0]
        markupsNodeTarget.GetNthControlPointPosition(0, firstNeedle)
        
        
        volumeArray = slicer.util.arrayFromVolume(volumeNode)
        volumeArray = (volumeArray > 0).astype(np.int32)
        TensorVolume = torch.tensor(volumeArray, device='cuda')

        w1 = 0.5 #weight for confidence ellipse and tumor overlap
        w2 = 0.5 #weight for iceball and tumor overlap
        overlapT = []
        overlapI = []
       
        if self.placement == 'Second placement':
            radii = (18,16.5,15) #iceball sizes
            self.directionvector1 = self.calculate_direction_miss(0)
            self.directionvector2 = None
            directionvectorTotal = self.total_direction_vector() #calculate direction vector of the miss
            #enerate points 1.5 cm from first needle location
            
            random_points, centerPoint = self.generateRandomPointsInVolume(firstNeedle, radius=10, num_points=40)

            #visualize points to see if it formed a correct sphere
            self.visualizeRandomPoints(random_points)  
            print(self.radii)
            visual_ellipse, ellipsoid_voxels= self.voxelize_ellipsoid(self.radii, volumeNode, directionvectorTotal) #calculate 95% confidence ellipse
            for point, firstNeedle in zip(random_points, centerPoint):
                overlapT_Ellipse = self.ellipsoid_and_tumor_overlap_percentage(volumeNode, TensorVolume, point, ellipsoid_voxels)
                
                #calculate properties for iceball
                direction_vector = np.array(firstNeedle) - np.array(point)
                direction_vector = direction_vector / np.linalg.norm(direction_vector) #calculate direction of mayor axis 

                center = (np.array(point) + np.array(firstNeedle)) / 2               
                center[2] = center[2] - 0.5*10 # needle tip is not the center but is lower on the needle shaft   
                visual_ice, iceball = self.voxelize_ellipsoid(radii, volumeNode, direction_vector)
                overlapiceball_tumor = self.ellipsoid_and_tumor_overlap_percentage(volumeNode, TensorVolume, center, iceball)
           
                #self.visualize_ellipsoid(volumeNode, iceball, center)
                #self.visualize_ellipsoid(volumeNode, tumor_voxels, center)
#                self.visualize_ellipsoid(volumeNode, ellipsoid_voxels, point)
#         
                overlapT.append(overlapT_Ellipse)
                overlapI.append(overlapiceball_tumor)
    
        if self.placement == 'Third placement':
            random_points = self.generateRandomPoint_pairs_InVolume(firstNeedle, radius=10, num_pairs=5)
            self.visualizeRandomPoint_pairs(random_points)
                   
            self.directionvector1 = self.calculate_direction_miss(0)
            self.directionvector2 = self.calculate_direction_miss(1)
            directionvectorTotal = self.total_direction_vector() #calculate direction vector of the miss

            for p in range(len(random_points)):
                
                visual_ellipse, ellipsoid_voxels= self.voxelize_ellipsoid(self.radii, volumeNode, directionvectorTotal) #calculate 95% confidence ellipse
                overlapT_Ellipse1 = self.ellipsoid_and_tumor_overlap_percentage(volumeNode, TensorVolume, random_points[p][1], ellipsoid_voxels)
                overlapT_Ellipse2 = self.ellipsoid_and_tumor_overlap_percentage(volumeNode, TensorVolume, random_points[p][2], ellipsoid_voxels)
                overlapT.append((overlapT_Ellipse1+overlapT_Ellipse2)/2)
                
                random_points0 = [random_points[p][0][0] ,random_points[p][0][1], random_points[p][0][2] - 0.5*10]
                random_points1 = [random_points[p][1][0] ,random_points[p][1][1], random_points[p][1][2] - 0.5*10]
                random_points2 = [random_points[p][2][0] ,random_points[p][2][1], random_points[p][2][2] - 0.5*10]
                overlapiceball_tumor = self.three_voxelized_ellipsoids(volumeNode, TensorVolume, random_points0,random_points1, random_points2)
                overlapI.append(overlapiceball_tumor)
        
        
        top_5 = sorted(overlapT, reverse=True)[:5]
        top_5_i = sorted(overlapI, reverse=True)[:5]
        low_5_i = sorted(overlapI, reverse=False)[:5]
        print(' highest overlap tumor' , top_5)
        print(' highest overlap iceball', top_5_i)
        print(' lowest overlap iceball', low_5_i)

        #Normalize fractions
        if max(overlapT) == min(overlapT):
            # If they are, set all normalized values to 0 (or another fixed value)
            normalized_set1 = [0 for _ in overlapT]
        else:
            normalized_set1 = [(f - min(overlapT)) / (max(overlapT) - min(overlapT)) for f in overlapT]
            
        if max(overlapI) == min(overlapI):
            # If they are, set all normalized values to 0 (or another fixed value)
            normalized_set2 = [0 for _ in overlapI]
        else:
            normalized_set2 = [(f - min(overlapI)) / (max(overlapI) - min(overlapI)) for f in overlapI]
        
        combined_fractions = [w1* f1 + w2*f2  for f1, f2 in zip(normalized_set1, normalized_set2)]
        normalized_combined = [(f - min(combined_fractions)) / (max(combined_fractions) - min(combined_fractions)) for f in combined_fractions]

        index = normalized_combined.index(max(normalized_combined))
        print(index)
        top_5index =np.argsort(normalized_combined)[-5:][::-1]
        print(top_5index)
        self.top5_points = [random_points[i] for i in top_5index]
        print('top points', self.top5_points)
        if self.placement == 'Second placement':
            point = random_points[index]
            cen = centerPoint[index]
            direction_vector = np.array(cen) - np.array(point)
            direction_vector = direction_vector / np.linalg.norm(direction_vector) #calculate direction of mayor axis 
            center = (np.array(point) + np.array(cen)) / 2               
            center[2] = center[2] - 0.5*radii[2] # needle tip is not the center but is lower on the needle shaft 
            
            visual_ice, iceball = self.voxelize_ellipsoid(radii, volumeNode, direction_vector)
            self.visualize_ellipsoid(volumeNode, visual_ice, center)
            self.visualize_ellipsoid(volumeNode, visual_ellipse, point)
            
            self.visualizeBestPoints(self.top5_points)
            
        if self.placement == 'Third placement':
            self.visualize_BestRandomPoint_pairs(random_points[index])
            self.visualize_ellipsoid(volumeNode, visual_ellipse, random_points[index][1])
            self.visualize_ellipsoid(volumeNode, visual_ellipse, random_points[index][2])

            random_points0 = [random_points[index][0][0] ,random_points[index][0][1], random_points[index][0][2] - 0.5*10]
            random_points1 = [random_points[index][1][0] ,random_points[index][1][1], random_points[index][1][2] - 0.5*10]
            random_points2 = [random_points[index][2][0] ,random_points[index][2][1], random_points[index][2][2] - 0.5*10]

            self.visualize_three_voxelized(volumeNode, random_points0,random_points1, random_points2)
            
            
        return random_points, normalized_combined
    
#%%
    def generateRandomPointsInVolume(self, centerPointRAS, radius, num_points):
        """
        Generate evenly distributed points on multiple circular layers around a center point in the XY plane,
        with each layer offset along the z-axis. Adjusts centerPointRAS up and down with each distance.
    
        Parameters:
            centerPointRAS: The initial center of the circle in RAS coordinates (x, y, z).
            radius: The radius of the circle in mm.
            num_points: Number of points to generate on the circumference of each circle.
    
        Returns:
            A numpy array of points on the circumference of each circle in RAS coordinates.
        """
        # List to store all points on each circular layer
        circle_points = []
        centers = []
        # Angle increment for each point to be evenly distributed around each circle
        angle_increment = 2 * np.pi / num_points
        
        # Z-offsets for each circle layer relative to the center Z coordinate
        distances = [-6, -4, -2, 0, 2, 4, 6]
        
        # Loop through each z-offset to create a new circle layer at that height
        for z_offset in distances:
            # Adjust centerPointRAS's z-coordinate for the current layer
            adjusted_center_z = centerPointRAS[2] + z_offset
            
            # Generate points around the circle in the XY plane at this adjusted z-coordinate
            for i in range(num_points):
                angle = i * angle_increment  # Angle for this point
                
                # Calculate the coordinates of the point on the circle
                x = centerPointRAS[0] + radius * np.cos(angle)
                y = centerPointRAS[1] + radius * np.sin(angle)
                z = adjusted_center_z
                
                # Append the point to the list
                circle_points.append([x, y, z])
                centers.append([centerPointRAS[0], centerPointRAS[1],adjusted_center_z])
        return np.array(circle_points), np.array(centers)


   
    def generateRandomPoint_pairs_InVolume(self, centerPointRAS, radius, num_pairs):
        """
        Generate symmetric pairs of points on the circumference of a circle in the XY plane.
        Each pair is positioned symmetrically around a central point with a 60-degree angle between them.
        
        Parameters:
            centerPointRAS: Tuple of (x, y, z) representing the center of the circle in RAS coordinates.
            radius: The radius of the circle in millimeters.
            num_pairs: The number of symmetric pairs to generate along the circumference.
        
        Returns:
            A list of tuples, where each tuple contains two points (each point is a numpy array of [x, y, z]),
            arranged symmetrically around each center point.
        """
        # List to store the point pairs
        point_pairs = []
        
        # Angle increment for each central point to be evenly distributed
        angle_increment = 2 * np.pi / num_pairs
        offset_angle = np.pi / 6  # Offset of 30 degrees for each side point in the pair
        # Z-offsets for each circle layer relative to the center Z coordinate
        distances = [-6, -4, -2, 0, 2, 4, 6]
        
        for z_offset in distances:
            # Adjust centerPointRAS's z-coordinate for the current layer
            adjusted_center_z = centerPointRAS[2] + z_offset
            for i in range(num_pairs):
                # Calculate the angle for the central point
                angle_center = i * angle_increment
                
                # Calculate coordinates of the left point (center - 30 degrees)
                angle_left = angle_center - offset_angle
                x_left = centerPointRAS[0] + radius * np.cos(angle_left)
                y_left = centerPointRAS[1] + radius * np.sin(angle_left)
                z_left = adjusted_center_z
                point_left = np.array([x_left, y_left, z_left])
                
                # Calculate coordinates of the right point (center + 30 degrees)
                angle_right = angle_center + offset_angle
                x_right = centerPointRAS[0] + radius * np.cos(angle_right)
                y_right = centerPointRAS[1] + radius * np.sin(angle_right)
                z_right = adjusted_center_z
                point_right = np.array([x_right, y_right, z_right])
                center = [centerPointRAS[0], centerPointRAS[1], adjusted_center_z]
                # Add the symmetric pair to the list
                point_pairs.append((center, point_left, point_right))
        
        return point_pairs
    
    def visualize_BestRandomPoint_pairs(self, random_points):
        """
        Visualizes given points
        """       
        randomPointsNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode2.SetName("Best Points")

        for point in random_points:
            print('best point', point)
            randomPointsNode2.AddControlPoint(point[0], point[1], point[2])

        # Set colors for the display nodes
        cluster1Display2 = randomPointsNode2.GetDisplayNode()
        cluster1Display2.SetTextScale(0)  # Hide the label text
        cluster1Display2.SetGlyphScale(2)  # Adjust point size
        cluster1Display2.SetSelectedColor(1,1, 0)  # Red for cluster 1
        
        
    def visualizeRandomPoint_pairs(self, random_points):
        """
        Visualizes given points
        """
        # Create a markups node for the random points
        randomPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode.SetName("Random Points")
        
        randomPointsNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode2.SetName("Random Points2")
        
        randomPointsNode3 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode3.SetName("Random Points3")
     
        for i in range(0, len(random_points)-2, 3):   
            for point in random_points[i]:           
                randomPointsNode.AddControlPoint(point[0], point[1], point[2])
            for point in random_points[i+1]:
                randomPointsNode2.AddControlPoint(point[0], point[1], point[2])
            for point in random_points[i+2]:
                randomPointsNode3.AddControlPoint(point[0], point[1], point[2])
                   
        # # Set colors for the display nodes
        cluster1Display = randomPointsNode.GetDisplayNode()
        cluster1Display.SetTextScale(0)  # Hide the label text
        cluster1Display.SetGlyphScale(1)  # Adjust point size
        cluster1Display.SetSelectedColor(0,1, 0)  # Red for cluster 1
        
                # # Set colors for the display nodes
        cluster1Display2 = randomPointsNode2.GetDisplayNode()
        cluster1Display2.SetTextScale(0)  # Hide the label text
        cluster1Display2.SetGlyphScale(1)  # Adjust point size
        cluster1Display2.SetSelectedColor(1,0, 0)  # Red for cluster 1
        
                # # Set colors for the display nodes
        cluster1Display3 = randomPointsNode3.GetDisplayNode()
        cluster1Display3.SetTextScale(0)  # Hide the label text
        cluster1Display3.SetGlyphScale(1)  # Adjust point size
        cluster1Display3.SetSelectedColor(0,0,1)  # Red for cluster 1
        
    def visualizeRandomPoints(self, random_points):
        """
        Visualizes given points
        """
        # Create a markups node for the random points
        randomPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode.SetName("Random Points")
        
        for point in random_points:
            randomPointsNode.AddControlPoint(point[0], point[1], point[2])  # Add each point as a control point
        # # Set colors for the display nodes
        cluster1Display = randomPointsNode.GetDisplayNode()
        cluster1Display.SetTextScale(0)  # Hide the label text
        cluster1Display.SetGlyphScale(1)  # Adjust point size
        cluster1Display.SetSelectedColor(0,1, 0)  # Red for cluster 1
        
    def visualizeBestPoints(self, random_points):
        """
        Visualizes given points
        """
        # Create a markups node for the random points
        randomPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        randomPointsNode.SetName("Best Points")
        
        for point in random_points:
            randomPointsNode.AddControlPoint(point[0], point[1], point[2])  # Add each point as a control point
        # # Set colors for the display nodes
        cluster1Display = randomPointsNode.GetDisplayNode()
        cluster1Display.SetTextScale(0)  # Hide the label text
        cluster1Display.SetGlyphScale(2)  # Adjust point size
        cluster1Display.SetSelectedColor(1,1, 0) # Red for cluster 1
#%%    
    def calculate_direction_miss(self, number):
        """
        Used to calculate the normalized direction vector between two points,
        needle and target
        Needle: gained from manual input through onSelectFirstNeedle
        Target: gained from a markupsnode loaded into 3d slicer
        """
        markupsNodeTarget = slicer.util.getNode('Needle')
        coord1 = [0.0, 0.0, 0.0]
        markupsNodeTarget.GetNthControlPointPosition(number, coord1)
        Needle = np.array(coord1)
        
        selectedMarkupsNode = self.markupsNodeSelector.currentNode()
        
        coord2 = [0.0, 0.0, 0.0]      
        selectedMarkupsNode.GetNthControlPointPosition(number, coord2)
        Planned = np.array(coord2)
        
        difference_vector = Needle - Planned
        # Calculate the magnitude (length) of the difference vector
        magnitude = np.linalg.norm(difference_vector)
        
        # Normalize the difference vector to get the direction
        direction_vector = difference_vector / magnitude
        
        return direction_vector
    
    def total_direction_vector(self):
        """
        Calculates total direction vector if multiple are given
        """
        if self.placement == 'Second placement':
            return self.directionvector1
        if self.placement == 'Third placement':
            mean_direction = (self.directionvector1 + self.directionvector2) / 2.0
            mean_direction = mean_direction / np.linalg.norm(mean_direction)
            
            return mean_direction
#%%
    def three_voxelized_ellipsoids(self, volumeNode,mask_tensor, position1, position2, position3):
        radii = (12,12,15)
        direction_vector = np.array([1,0,0])
        ellipsoid_voxels,nee = self.voxelize_ellipsoid(radii, volumeNode, direction_vector)        

        overlap = self.ellipsoid_and_tumor_overlap_percentage2(volumeNode, mask_tensor, position1, position2, position3, ellipsoid_voxels)
        
        return overlap
    
    def visualize_three_voxelized(self, volumeNode, position1, position2, position3):
        radii = (12,12,15)
        direction_vector = np.array([1,0,0])
        ellipsoid_voxels,nee = self.voxelize_ellipsoid(radii, volumeNode, direction_vector)        
        spacing = volumeNode.GetSpacing()
        
        # Convert ellipsoid_voxels and center from voxel to RAS coordinates
        center_tensor = torch.tensor(position1, dtype=torch.float32, device='cuda')
        ras_voxellized1 = (ellipsoid_voxels * torch.tensor(spacing, device='cuda')) + center_tensor
        
        # Convert ellipsoid_voxels and center from voxel to RAS coordinates
        center_tensor = torch.tensor(position2, dtype=torch.float32, device='cuda')
    
        ras_voxellized2 = (ellipsoid_voxels * torch.tensor(spacing, device='cuda')) + center_tensor
        
        # Convert ellipsoid_voxels and center from voxel to RAS coordinates
        center_tensor = torch.tensor(position3, dtype=torch.float32, device='cuda')
        ras_voxellized3 = (ellipsoid_voxels * torch.tensor(spacing, device='cuda')) + center_tensor
        combined_ellipsoid = torch.cat([ras_voxellized1, ras_voxellized2, ras_voxellized3], dim=0)
        self.visualize_ellipsoid2(volumeNode, combined_ellipsoid)
        
    def voxelize_ellipsoid(self, radii, volumeNode, direction_vector):
        """
        Uses the ellipsoids radius and direction vector to create a tensor voxel representation of the ellipse
        """
        spacing = volumeNode.GetSpacing()
        
        # Compute voxel radii
        rx = int(np.ceil(radii[0] / spacing[0]))  # Round up to ensure we cover the radius
        ry = int(np.ceil(radii[1] / spacing[1]))
        rz = int(np.ceil(radii[2] / spacing[2]))
        
        # Create a grid of points within the bounding box for the ellipsoid
        x = np.arange(-rx, rx + 1)
        y = np.arange(-ry, ry + 1)
        z = np.arange(-rz, rz + 1)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        
        # Use ellipsoid equation to get voxel points inside the ellipsoid
        ellipsoid_voxels = (grid_x**2 / rx**2 + grid_y**2 / ry**2 + grid_z**2 / rz**2) <= 1
        
        # Get the indices of the voxels that are inside the ellipsoid
        ellipsoid_indices = np.column_stack(np.where(ellipsoid_voxels))
        
        # Shift the indices to the center of the ellipsoid
        ellipsoid_indices -= np.array([rx, ry, rz])
        
        
        # Define the rotation matrix using the direction vector
        # Assuming that the first principal component aligns with the x-axis
        principal_axis = np.array([1, 0, 0])  # This can be any fixed axis to start with
        rotation_axis = np.cross(principal_axis, direction_vector)
        
        if np.linalg.norm(rotation_axis) > 0:  # If the rotation axis is not zero
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(principal_axis, direction_vector), -1.0, 1.0))  # Angle between the two vectors
        else:
            angle = 0  # No rotation needed if the two vectors are aligned
        
        # Create the rotation matrix
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        rotation_matrix = np.array([
        [cos_angle + rotation_axis[0]**2 * (1 - cos_angle), rotation_axis[0]*rotation_axis[1]*(1 - cos_angle) - rotation_axis[2]*sin_angle, rotation_axis[0]*rotation_axis[2]*(1 - cos_angle) + rotation_axis[1]*sin_angle],
        [rotation_axis[1]*rotation_axis[0]*(1 - cos_angle) + rotation_axis[2]*sin_angle, cos_angle + rotation_axis[1]**2 * (1 - cos_angle), rotation_axis[1]*rotation_axis[2]*(1 - cos_angle) - rotation_axis[0]*sin_angle],
        [rotation_axis[2]*rotation_axis[0]*(1 - cos_angle) - rotation_axis[1]*sin_angle, rotation_axis[2]*rotation_axis[1]*(1 - cos_angle) + rotation_axis[0]*sin_angle, cos_angle + rotation_axis[2]**2 * (1 - cos_angle)]
        ])
        
        # Rotate the ellipsoid indices
        rotated_indices = ellipsoid_indices @ rotation_matrix.T  # Apply rotation
        rotated_indices_long = torch.tensor(rotated_indices, dtype=torch.long, device='cuda')
        rotated_indices_visualization = torch.tensor(rotated_indices, dtype=torch.float32, device='cuda')
        # Convert to tensor and ensure proper type
        return rotated_indices_visualization, rotated_indices_long
    
    def ellipsoid_and_tumor_overlap_percentage(self, volumeNode, mask_tensor, ellipsoid_point, ellipsoid_voxels):
        """
        Compares the voxels from the ellipsoid with the tumor mask, to define the overlap score
        """

        # Ensure binary mask tensor on GPU
        mask_tensor = (mask_tensor > 0).int().to('cuda')
        
        # Convert ellipsoid center point to IJK and apply voxel scaling
        Ijkpoint = self.rasToIJK(volumeNode, ellipsoid_point)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels = torch.round(shifted_ellipsoid_voxels).long()

        # In-bounds filtering for shifted ellipsoid voxels
        in_bounds = (
            (shifted_ellipsoid_voxels[:, 0] >= 0) & 
            (shifted_ellipsoid_voxels[:, 0] < mask_tensor.shape[0]) & 
            (shifted_ellipsoid_voxels[:, 1] >= 0) & 
            (shifted_ellipsoid_voxels[:, 1] < mask_tensor.shape[1]) & 
            (shifted_ellipsoid_voxels[:, 2] >= 0) & 
            (shifted_ellipsoid_voxels[:, 2] < mask_tensor.shape[2])
        )
        valid_ellipsoid_voxels = shifted_ellipsoid_voxels[in_bounds].long()

        # Count voxels within tumor mask and ellipsoid
        inside_mask_count = (mask_tensor[valid_ellipsoid_voxels[:, 0], valid_ellipsoid_voxels[:, 1], valid_ellipsoid_voxels[:, 2]]>0).sum().item()

        # Tumor Size for overlap calculation
        TumorSize = mask_tensor.sum().item()
        
        # Fraction of ellipsoid voxels within tumor mask
        fraction_in_tumor = (inside_mask_count / TumorSize) if TumorSize > 0 else 0.0

        return fraction_in_tumor

    def ellipsoid_and_tumor_overlap_percentage2(self, volumeNode, mask_tensor, position1, position2, position3, ellipsoid_voxels):
        """
        Compares the voxels from the ellipsoid with the tumor mask, to define the overlap score
        """
        # Ensure binary mask tensor on GPU
        mask_tensor = (mask_tensor > 0).int().to('cuda')
        
        # Convert ellipsoid center point to IJK and apply voxel scaling
        Ijkpoint = self.rasToIJK(volumeNode, position1)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels1 = torch.round(shifted_ellipsoid_voxels).long()
        
        Ijkpoint = self.rasToIJK(volumeNode, position2)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels2 = torch.round(shifted_ellipsoid_voxels).long()

        Ijkpoint = self.rasToIJK(volumeNode, position3)
        ellipsoid_point_tensor = torch.tensor(Ijkpoint, device='cuda')

        # Get voxel size (spacing) from the volume node
        voxel_size = torch.tensor(volumeNode.GetSpacing(), device='cuda')
        
        # Scale ellipsoid voxels by the voxel size
        ellipsoid_voxels_scaled = ellipsoid_voxels * voxel_size  # Scale to mm

        # Shift ellipsoid voxels to align with the volume's voxel grid
        shifted_ellipsoid_voxels = ellipsoid_voxels_scaled + ellipsoid_point_tensor
        shifted_ellipsoid_voxels3 = torch.round(shifted_ellipsoid_voxels).long()        

        combined_ellipsoid = torch.cat([shifted_ellipsoid_voxels1, shifted_ellipsoid_voxels2, shifted_ellipsoid_voxels3], dim=0)

        
        # In-bounds filtering for shifted ellipsoid voxels
        in_bounds = (
            (combined_ellipsoid[:, 0] >= 0) & 
            (combined_ellipsoid[:, 0] < mask_tensor.shape[0]) & 
            (combined_ellipsoid[:, 1] >= 0) & 
            (combined_ellipsoid[:, 1] < mask_tensor.shape[1]) & 
            (combined_ellipsoid[:, 2] >= 0) & 
            (combined_ellipsoid[:, 2] < mask_tensor.shape[2])
        )
        valid_ellipsoid_voxels = combined_ellipsoid[in_bounds].long()
     
        # Count voxels within tumor mask and ellipsoid
        inside_mask_count = (mask_tensor[valid_ellipsoid_voxels[:, 0], valid_ellipsoid_voxels[:, 1], valid_ellipsoid_voxels[:, 2]]>0).sum().item()

        # Tumor Size for overlap calculation
        TumorSize = mask_tensor.sum().item()
        
        # Fraction of ellipsoid voxels within tumor mask
        fraction_in_tumor = (inside_mask_count / TumorSize) if TumorSize > 0 else 0.0

        return fraction_in_tumor
     
    def visualize_ellipsoid(self, volumeNode, ellipsoid_voxels, center):
        """
        Visualizes an ellipse in 3D slicer
        """
        # Get voxel spacing
        spacing = volumeNode.GetSpacing()
        
        # Convert ellipsoid_voxels and center from voxel to RAS coordinates
        center_tensor = torch.tensor(center, dtype=torch.float32, device='cuda')
    
        ellipsoid_points = (ellipsoid_voxels * torch.tensor(spacing, device='cuda')) + center_tensor
    
        # Convert to numpy for visualization
        ellipse_points = ellipsoid_points.cpu().numpy()  # Convert to numpy for visualization
   
        vtk_points = vtk.vtkPoints()
        for point in ellipse_points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
    
        # Create a vtkPolyData object to store the points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
    
        # Create a surface from the points using Delaunay triangulation
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        delaunay.Update()
        surfaceFilter = vtk.vtkDataSetSurfaceFilter()
        surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
        surfaceFilter.Update()
        polydata = surfaceFilter.GetOutput()
    
        # Create a model node and assign the polydata to it
        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        model_node.SetAndObservePolyData(polydata)
    
        # Create a segmentation node and import the model node into it
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentationNode)
    
        # Set color and display properties
        segmentationNode.GetDisplayNode().SetColor(0, 1, 0)  # Set color to green
        segmentationNode.GetDisplayNode().SetVisibility3D(True)  # Show in 3D view
        segmentationNode.GetDisplayNode().SetOpacity(0.4)
        segmentationNode.GetDisplayNode().SetVisibility2DFill(False)  # Fill in 2D slice views
    
        # Reset the 3D view's focal point to focus on the ellipsoid
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()         

    def visualize_ellipsoid2(self, volumeNode, ellipsoid_voxels):
        """
        Visualizes an ellipse in 3D slicer
        """

    
        # Convert to numpy for visualization
        ellipse_points = ellipsoid_voxels.cpu().numpy()  # Convert to numpy for visualization
   
        vtk_points = vtk.vtkPoints()
        for point in ellipse_points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
    
        # Create a vtkPolyData object to store the points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
    
        # Create a surface from the points using Delaunay triangulation
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(polydata)
        delaunay.Update()
        surfaceFilter = vtk.vtkDataSetSurfaceFilter()
        surfaceFilter.SetInputConnection(delaunay.GetOutputPort())
        surfaceFilter.Update()
        polydata = surfaceFilter.GetOutput()
    
        # Create a model node and assign the polydata to it
        model_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        model_node.SetAndObservePolyData(polydata)
    
        # Create a segmentation node and import the model node into it
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentationNode)
    
        # Set color and display properties
        segmentationNode.GetDisplayNode().SetColor(0, 1, 0)  # Set color to green
        segmentationNode.GetDisplayNode().SetVisibility3D(True)  # Show in 3D view
        segmentationNode.GetDisplayNode().SetOpacity(0.4)
        segmentationNode.GetDisplayNode().SetVisibility2DFill(False)  # Fill in 2D slice views
    
        # Reset the 3D view's focal point to focus on the ellipsoid
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()     
        
    def rasToIJK(self, volumeNode, ras_point):
        # Get spacing, origin, and direction matrix
        spacing = volumeNode.GetSpacing()  # (0.9375, 0.9375, 3.6)
        origin = volumeNode.GetOrigin()    # (90.583, 117.562, 8.79922)
        directionMatrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASDirectionMatrix(directionMatrix)

        # Subtract the origin from the RAS point to account for the translation
        ras_point_adjusted = np.array(ras_point) - np.array(origin)
    
        # Get the inverse of the direction matrix for transforming RAS to IJK
        directionMatrix.Invert()
    
        # Apply the direction matrix
        ras_homogeneous = np.array([ras_point_adjusted[0], ras_point_adjusted[1], ras_point_adjusted[2], 1])
        ijk_homogeneous = directionMatrix.MultiplyPoint(ras_homogeneous)
    
        # Divide by spacing to account for non-uniform voxel size
        ijk_coords = np.array([ijk_homogeneous[2] / spacing[2],
                               ijk_homogeneous[1] / spacing[1],
                               ijk_homogeneous[0] / spacing[0]])
    
        # Round to get integer IJK coordinates
        ijk_coords = np.round(ijk_coords).astype(int)
    
        return ijk_coords
    
#%%
    def onAI_iceball(self):
        bestPoint = self.top5_points[0] #top5 stores the best 5 points, and 0 is the best      
        #determine first needle location
        markupsNodeTarget = slicer.util.getNode('Needle')
        firstNeedle = [0.0, 0.0, 0.0]
        markupsNodeTarget.GetNthControlPointPosition(0, firstNeedle) #retrieves location of first needle
        
        volume = self.volumeIceSelector.currentNode() #load volume to calculate iceball on
        print(volume)