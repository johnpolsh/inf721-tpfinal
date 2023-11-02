package com.jopzoli.objectdetection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.Navigation

class PermissionsFragment : Fragment() {
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            if (isGranted) {
                Toast.makeText(context, resources.getString(R.string.camera_permission_granted), Toast.LENGTH_LONG).show()
                navigateToCamera()
            } else {
                Toast.makeText(context, resources.getString(R.string.camera_permission_denied), Toast.LENGTH_LONG).show()
                ActivityCompat.finishAffinity(this.activity as MainActivity);
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) -> {
                navigateToCamera()
            }
            else -> {
                requestPermissionLauncher.launch(
                    Manifest.permission.CAMERA)
            }
        }
    }

    private fun navigateToCamera() {
        lifecycleScope.launchWhenStarted {
            Navigation.findNavController(requireActivity(), R.id.nav_host_fragment_content_main)
                .navigate(R.id.action_permissions_to_camera)
        }
    }

    companion object {
        private val _cPERMISSIONSREQUIRED0 = arrayOf(Manifest.permission.CAMERA)

        fun hasPermissions(context: Context) = _cPERMISSIONSREQUIRED0.all {
            ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}